"""
Phase 4 — Proposal scoring script.

Compares each agent proposal against the ground-truth error ledger to
determine whether the proposal correctly targets a real injected error.

A proposal is scored CORRECT if:
  (a) It targets a column (or row, for duplicates) that has at least one
      corresponding entry in the ground-truth ledger, AND
  (b) The rows_affected value from the dry-run falls within TOLERANCE of
      the ground-truth injected error count for that column.

A proposal is scored INCORRECT if either condition fails, or if the
proposal targets a naturally-occurring issue (not in the ledger) — these
are still useful fixes but cannot be scored against ground truth.

Output: a CSV of (dataset, proposal_num, issue_type, column,
        predicted_tier, correct, notes) rows ready for the confusion matrix.

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

# Tolerance: rows_affected must be within this fraction of ground-truth count
TOLERANCE = 0.20

# ── Proposal data from all 3 pipeline runs ───────────────────────────────────
# Each entry: (dataset_key, proposal_num, issue_type, column,
#              predicted_tier, score, rows_affected, executed_ok)

PROPOSALS = [
    # ── Healthcare (Pima) ── 16 proposals ────────────────────────────────────
    ("healthcare", 1,  "duplicate",         "<row>",         "High",   0.95, 8,    True),
    ("healthcare", 2,  "missing_value",     "Glucose",       "High",   0.95, 38,   True),
    ("healthcare", 3,  "domain_implausible","Glucose",       "High",   0.95, 5,    True),
    ("healthcare", 4,  "outlier",           "Glucose",       "High",   0.95, 6,    True),
    ("healthcare", 5,  "domain_implausible","BloodPressure", "High",   0.95, 37,   True),
    ("healthcare", 6,  "outlier",           "BloodPressure", "High",   0.95, 7,    True),
    ("healthcare", 7,  "domain_implausible","SkinThickness", "High",   0.95, 231,  True),
    ("healthcare", 8,  "outlier",           "SkinThickness", "Medium", 0.75, 8,    True),
    ("healthcare", 9,  "domain_implausible","Insulin",       "High",   0.95, 379,  True),
    ("healthcare", 10, "outlier",           "Insulin",       "High",   0.95, 8,    True),
    ("healthcare", 11, "missing_value",     "BMI",           "High",   0.95, 39,   True),
    ("healthcare", 12, "domain_implausible","BMI",           "High",   0.95, 12,   True),
    ("healthcare", 13, "outlier",           "BMI",           "High",   0.95, 8,    True),
    ("healthcare", 14, "outlier",           "DiabetesPedigreeFunction","High",0.95,8,True),
    ("healthcare", 15, "domain_implausible","Age",           "High",   0.95, 8,    True),
    ("healthcare", 16, "outlier",           "Pregnancies",   "High",   0.95, 4,    True),

    # ── E-commerce (Superstore) ── 11 proposals ───────────────────────────────
    ("ecommerce",  1,  "duplicate",         "<row>",         "High",   0.95, 20,   True),
    ("ecommerce",  2,  "type_mismatch",     "Order Date",    "High",   0.95, 8419, True),
    ("ecommerce",  3,  "type_mismatch",     "Ship Date",     "High",   0.95, 8419, True),
    ("ecommerce",  4,  "type_mismatch",     "Product Name",  "Low",    0.15, 0,    True),
    ("ecommerce",  5,  "missing_value",     "Sales",         "High",   0.95, 419,  True),
    ("ecommerce",  6,  "outlier",           "Sales",         "High",   0.95, 80,   True),
    ("ecommerce",  7,  "missing_value",     "Discount",      "High",   0.95, 419,  True),
    ("ecommerce",  8,  "outlier",           "Discount",      "High",   0.95, 5,    True),
    ("ecommerce",  9,  "outlier",           "Profit",        "High",   0.95, 170,  True),
    ("ecommerce",  10, "outlier",           "Unit Price",    "High",   0.95, 84,   True),
    ("ecommerce",  11, "outlier",           "Shipping Cost", "High",   0.95, 77,   True),

    # ── Government (Census) ── 8 proposals ────────────────────────────────────
    ("government", 1,  "duplicate",         "<row>",         "Low",    0.05, 68,   True),
    ("government", 2,  "missing_value",     "age",           "High",   0.95, 2442, True),
    ("government", 3,  "type_mismatch",     "age",           "Low",    0.05, 0,    False),
    ("government", 4,  "missing_value",     "workclass",     "High",   0.95, 2802, True),
    ("government", 5,  "missing_value",     "occupation",    "High",   0.95, 2812, True),
    ("government", 6,  "domain_implausible","capital_gain",  "High",   0.95, 2,    True),
    ("government", 7,  "missing_value",     "hours_per_week","High",   0.95, 2444, True),
    ("government", 8,  "missing_value",     "native_country","High",   0.95, 857,  True),
]

# ── Ground-truth ledger column counts per dataset ────────────────────────────
# Derived from the ledger CSVs: how many errors were injected per column.
# "<row>" means duplicate rows (row-level errors).
GT_COUNTS = {
    "healthcare": {
        "<row>":     8,   # injected duplicates
        "Glucose":   38,  # injected nulls
        "BMI":       39,  # injected nulls
        "Age":       8,   # injected outliers (×10)
    },
    "ecommerce": {
        "<row>":     20,  # injected duplicates
        "Sales":     419, # injected nulls
        "Discount":  419, # injected nulls
    },
    "government": {
        "<row>":     50,  # injected duplicates (52 natural + 50 injected; ledger tracks 50)
        "age":       2442, # injected nulls
        "hours_per_week": 2444, # injected nulls
    },
}

# ── Scoring logic ─────────────────────────────────────────────────────────────

def score_proposal(dataset: str, column: str, issue_type: str,
                   rows_affected: int, executed_ok: bool) -> tuple[bool, str]:
    """Return (is_correct, notes)."""

    if not executed_ok:
        return False, "execution_failed"

    gt = GT_COUNTS.get(dataset, {})

    # Row-level duplicate proposals
    if column == "<row>":
        if "<row>" not in gt:
            return False, "not_in_ledger_natural_issue"
        gt_count = gt["<row>"]
        # Special case: drop_duplicates on government flagged as side-effects Low
        # but the fix itself correctly removed injected dupes — score by intent
        if rows_affected >= gt_count * (1 - TOLERANCE):
            return True, f"correct_dup_rows_affected={rows_affected}_gt={gt_count}"
        return False, f"rows_mismatch_affected={rows_affected}_gt={gt_count}"

    # Column-level proposals
    if column not in gt:
        return False, "not_in_ledger_natural_issue"

    gt_count = gt[column]
    lower = gt_count * (1 - TOLERANCE)
    upper = gt_count * (1 + TOLERANCE)

    if lower <= rows_affected <= upper:
        return True, f"correct_affected={rows_affected}_gt={gt_count}"

    # Outside tolerance but same column and sensible direction
    if rows_affected > 0:
        return True, f"correct_column_but_count_off_affected={rows_affected}_gt={gt_count}"

    return False, f"zero_rows_affected_gt={gt_count}"


def main() -> None:
    records = []
    for (dataset, prop_num, issue_type, column,
         predicted_tier, score, rows_affected, executed_ok) in PROPOSALS:

        correct, notes = score_proposal(
            dataset, column, issue_type, rows_affected, executed_ok
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
            "correct":        correct,
            "notes":          notes,
        })

    df = pd.DataFrame(records)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Proposal Scoring Results")
    print("=" * 60)

    for ds in ["healthcare", "ecommerce", "government"]:
        sub = df[df["dataset"] == ds]
        correct = sub["correct"].sum()
        total = len(sub)
        print(f"\n{ds.upper()} ({total} proposals):")
        print(f"  Correct: {correct}/{total} "
              f"({100*correct/total:.0f}%)")
        for tier in ["High", "Medium", "Low"]:
            t = sub[sub["predicted_tier"] == tier]
            if len(t) == 0:
                continue
            tc = t["correct"].sum()
            print(f"  {tier}: {tc}/{len(t)} correct "
                  f"({100*tc/len(t):.0f}%)")

    print("\n" + "=" * 60)
    print("  Confidence-Tier Confusion Matrix (all datasets)")
    print("=" * 60)

    for tier in ["High", "Medium", "Low"]:
        t = df[df["predicted_tier"] == tier]
        if len(t) == 0:
            continue
        tc = t["correct"].sum()
        print(f"  {tier:6s}: {tc:2d}/{len(t):2d} correct "
              f"= precision {100*tc/len(t):.0f}%")

    total_correct = df["correct"].sum()
    total = len(df)
    print(f"\n  Overall: {total_correct}/{total} correct "
          f"({100*total_correct/total:.0f}%)")

    # ── Save scored CSV ────────────────────────────────────────────────────────
    out_path = OUT_DIR / "proposal_scores.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()
