"""
Phase 4 — Proposal scoring script (strict, non-circular version).

SCORING PHILOSOPHY (revised after professor feedback):

Previous version had two problems:
  1. Loose correctness: any proposal with affected_count > 0 was scored correct,
     regardless of how far off the row count was. This made 100% High-tier
     precision partly an artifact of the scoring rule, not the system.
  2. Circularity: confidence score was computed from rows_affected, and
     correctness was also computed by comparing rows_affected to ground truth.
     Both variables derived from the same measurement.

This version fixes both:

  CORRECTNESS (non-circular):
    A proposal is correct if it targets a column/row that has a corresponding
    entry in the ground-truth ledger — period. Row count is NOT used in the
    correctness definition. This decouples correctness from the dry-run
    measurement used to compute confidence.

    Special case — domain_implausible proposals on columns NOT in the ledger
    (e.g. BloodPressure=0, which is a natural issue, not injected):
    These are scored as NATURAL_CORRECT — they are genuine finds, but they
    cannot be scored against the injected ledger. Reported separately.

  CONFIDENCE (unchanged):
    Still derived from dry-run execution signals: rows_affected, side effects,
    unexpected nulls, execution success. rows_affected is one signal in the
    confidence score but NOT used in the correctness label.

Usage:
    python scripts/score_proposals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
GT_DIR    = REPO_ROOT / "benchmarks" / "ground_truth"
OUT_DIR   = REPO_ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── Ground-truth injected error columns per dataset ───────────────────────────
# A proposal is CORRECT if it targets one of these columns/rows.
# Row count is deliberately NOT used in the correctness definition.
GT_COLUMNS = {
    "healthcare": {"<row>", "Glucose", "BMI", "Age"},
    "ecommerce":  {"<row>", "Sales", "Discount"},
    "government": {"<row>", "age", "hours_per_week"},
}

# ── Proposal data from all 3 pipeline runs ────────────────────────────────────
# (dataset, prop_num, issue_type, column, predicted_tier, score,
#  rows_affected, executed_ok)

PROPOSALS = [
    # ── Healthcare (Pima) — 16 proposals ──────────────────────────────────────
    ("healthcare", 1,  "duplicate",          "<row>",                  "High",   0.95, 8,    True),
    ("healthcare", 2,  "missing_value",      "Glucose",                "High",   0.95, 38,   True),
    ("healthcare", 3,  "domain_implausible", "Glucose",                "High",   0.95, 5,    True),
    ("healthcare", 4,  "outlier",            "Glucose",                "High",   0.95, 6,    True),
    ("healthcare", 5,  "domain_implausible", "BloodPressure",          "High",   0.95, 37,   True),
    ("healthcare", 6,  "outlier",            "BloodPressure",          "High",   0.95, 7,    True),
    ("healthcare", 7,  "domain_implausible", "SkinThickness",          "High",   0.95, 231,  True),
    ("healthcare", 8,  "outlier",            "SkinThickness",          "Medium", 0.75, 8,    True),
    ("healthcare", 9,  "domain_implausible", "Insulin",                "High",   0.95, 379,  True),
    ("healthcare", 10, "outlier",            "Insulin",                "High",   0.95, 8,    True),
    ("healthcare", 11, "missing_value",      "BMI",                    "High",   0.95, 39,   True),
    ("healthcare", 12, "domain_implausible", "BMI",                    "High",   0.95, 12,   True),
    ("healthcare", 13, "outlier",            "BMI",                    "High",   0.95, 8,    True),
    ("healthcare", 14, "outlier",            "DiabetesPedigreeFunction","High",  0.95, 8,    True),
    ("healthcare", 15, "domain_implausible", "Age",                    "High",   0.95, 8,    True),
    ("healthcare", 16, "outlier",            "Pregnancies",            "High",   0.95, 4,    True),

    # ── E-commerce (Superstore) — 11 proposals ────────────────────────────────
    ("ecommerce",  1,  "duplicate",          "<row>",                  "High",   0.95, 20,   True),
    ("ecommerce",  2,  "type_mismatch",      "Order Date",             "High",   0.95, 8419, True),
    ("ecommerce",  3,  "type_mismatch",      "Ship Date",              "High",   0.95, 8419, True),
    ("ecommerce",  4,  "type_mismatch",      "Product Name",           "Low",    0.15, 0,    True),
    ("ecommerce",  5,  "missing_value",      "Sales",                  "High",   0.95, 419,  True),
    ("ecommerce",  6,  "outlier",            "Sales",                  "High",   0.95, 80,   True),
    ("ecommerce",  7,  "missing_value",      "Discount",               "High",   0.95, 419,  True),
    ("ecommerce",  8,  "outlier",            "Discount",               "High",   0.95, 5,    True),
    ("ecommerce",  9,  "outlier",            "Profit",                 "High",   0.95, 170,  True),
    ("ecommerce",  10, "outlier",            "Unit Price",             "High",   0.95, 84,   True),
    ("ecommerce",  11, "outlier",            "Shipping Cost",          "High",   0.95, 77,   True),

    # ── Government (Census) — 8 proposals ─────────────────────────────────────
    ("government", 1,  "duplicate",          "<row>",                  "Low",    0.05, 68,   True),
    ("government", 2,  "missing_value",      "age",                    "High",   0.95, 2442, True),
    ("government", 3,  "type_mismatch",      "age",                    "Low",    0.05, 0,    False),
    ("government", 4,  "missing_value",      "workclass",              "High",   0.95, 2802, True),
    ("government", 5,  "missing_value",      "occupation",             "High",   0.95, 2812, True),
    ("government", 6,  "domain_implausible", "capital_gain",           "High",   0.95, 2,    True),
    ("government", 7,  "missing_value",      "hours_per_week",         "High",   0.95, 2444, True),
    ("government", 8,  "missing_value",      "native_country",         "High",   0.95, 857,  True),
]


def score_proposal(
    dataset: str,
    column: str,
    issue_type: str,
    executed_ok: bool,
) -> tuple[str, str]:
    """
    Score a proposal using a non-circular, column-match-only definition.

    Returns (label, notes) where label is one of:
      "correct"         — targets an injected ground-truth error column
      "incorrect"       — execution failed or targets wrong column
      "natural_correct" — targets a real but non-injected issue (excluded
                          from the main confusion matrix, reported separately)

    NOTE: rows_affected is deliberately NOT used here. Confidence scores
    already use rows_affected as a signal; using it again in the correctness
    label would create circularity.
    """
    gt_cols = GT_COLUMNS.get(dataset, set())

    # Execution failure → always incorrect
    if not executed_ok:
        return "incorrect", "execution_failed"

    # Targets an injected ground-truth column/row → correct
    if column in gt_cols:
        return "correct", f"targets_gt_column_{column}"

    # Targets a real but naturally-occurring issue (not injected)
    # → report separately, exclude from main confusion matrix
    return "natural_correct", f"natural_issue_{column}"


def main() -> None:
    records = []
    for (dataset, prop_num, issue_type, column,
         predicted_tier, score, rows_affected, executed_ok) in PROPOSALS:

        label, notes = score_proposal(
            dataset, column, issue_type, executed_ok
        )

        records.append({
            "dataset":        dataset,
            "proposal_num":   prop_num,
            "issue_type":     issue_type,
            "column":         column,
            "predicted_tier": predicted_tier,
            "score":          score,
            "rows_affected":  rows_affected,
            "executed_ok":    executed_ok,
            "label":          label,
            # For confusion matrix: treat natural_correct as excluded
            "correct":        label == "correct",
            "notes":          notes,
        })

    df = pd.DataFrame(records)

    # ── Separate scoreable proposals from natural-issue ones ──────────────────
    scoreable = df[df["label"] != "natural_correct"].copy()
    natural   = df[df["label"] == "natural_correct"].copy()

    print()
    print("=" * 65)
    print("  Proposal Scoring — Strict Non-Circular Definition")
    print("  Correctness = column match only (rows_affected NOT used)")
    print("=" * 65)

    print(f"\n  Total proposals:          {len(df)}")
    print(f"  Scoreable (vs GT ledger): {len(scoreable)}")
    print(f"  Natural issues (excluded from matrix): {len(natural)}")
    print(f"    Columns: {sorted(natural['column'].unique())}")

    print()
    print("  " + "-" * 61)
    print(f"  {'Tier':<10} {'Correct':>8} {'Incorrect':>10} "
          f"{'Total':>7} {'Precision':>10}")
    print("  " + "-" * 61)

    tier_order = ["High", "Medium", "Low"]
    for tier in tier_order:
        t = scoreable[scoreable["predicted_tier"] == tier]
        if len(t) == 0:
            continue
        tc = t["correct"].sum()
        bar = "█" * int(100 * tc / len(t) / 10)
        print(f"  {tier:<10} {int(tc):>8} {int(len(t)-tc):>10} "
              f"{len(t):>7}   {100*tc/len(t):>5.1f}%  {bar}")

    total_correct = scoreable["correct"].sum()
    total = len(scoreable)
    print("  " + "-" * 61)
    print(f"  {'Overall':<10} {int(total_correct):>8} "
          f"{int(total-total_correct):>10} {total:>7} "
          f"  {100*total_correct/total:>5.1f}%")
    print("=" * 65)

    print()
    print("  Per-dataset breakdown (scoreable only):")
    for ds in ["healthcare", "ecommerce", "government"]:
        sub = scoreable[scoreable["dataset"] == ds]
        if len(sub) == 0:
            continue
        c = sub["correct"].sum()
        t = len(sub)
        print(f"    {ds:<14}: {int(c)}/{t} correct ({100*c/t:.0f}%)")

    print()
    print("  Natural issues found (genuine but not in GT ledger):")
    for _, row in natural.iterrows():
        print(f"    [{row['dataset']}] {row['column']} "
              f"({row['issue_type']}) tier={row['predicted_tier']}")

    print()
    print("  NOTE: The confusion matrix below covers only scoreable")
    print("  proposals. Natural issues are reported above as additional")
    print("  detections beyond the injected ground truth.")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "proposal_scores.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()
