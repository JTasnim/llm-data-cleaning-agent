"""
Phase 4 — Confidence-tier confusion matrix (Section 7.2.1 of the proposal).

Reads proposal_scores.csv produced by scripts/score_proposals.py and:
  1. Prints the formal confusion matrix table
  2. Computes precision-at-tier for High / Medium / Low
  3. Saves a bar chart of precision-at-tier to outputs/

Usage:
    python src/evaluation/confusion_matrix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_CSV = REPO_ROOT / "outputs" / "proposal_scores.csv"
OUT_DIR    = REPO_ROOT / "outputs"

TIER_ORDER  = ["High", "Medium", "Low"]
TIER_COLORS = {"High": "#28A745", "Medium": "#FFC107", "Low": "#DC3545"}


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate predicted tier vs. actual correctness."""
    matrix = pd.crosstab(
        df["predicted_tier"],
        df["correct"],
        rownames=["Predicted tier"],
        colnames=["Actually correct"]
    ).reindex(TIER_ORDER, fill_value=0)
    matrix.columns = [str(c) for c in matrix.columns]
    if "True"  not in matrix.columns: matrix["True"]  = 0
    if "False" not in matrix.columns: matrix["False"] = 0
    matrix = matrix[["True", "False"]]
    matrix.columns = ["Correct", "Incorrect"]
    matrix["Total"]     = matrix["Correct"] + matrix["Incorrect"]
    matrix["Precision"] = (matrix["Correct"] / matrix["Total"] * 100).round(1)
    return matrix


def print_matrix(matrix: pd.DataFrame, df: pd.DataFrame) -> None:
    print()
    print("=" * 62)
    print("  Confidence-Tier Confusion Matrix")
    print("  LLM-Powered Data Cleaning Agent — Phase 4 Evaluation")
    print("=" * 62)
    print(f"  {'Tier':<10} {'Correct':>8} {'Incorrect':>10} "
          f"{'Total':>7} {'Precision':>10}")
    print("  " + "-" * 58)
    for tier in TIER_ORDER:
        if tier not in matrix.index:
            continue
        row = matrix.loc[tier]
        bar = "█" * int(row["Precision"] / 10)
        print(f"  {tier:<10} {int(row['Correct']):>8} "
              f"{int(row['Incorrect']):>10} {int(row['Total']):>7} "
              f"  {row['Precision']:>5.1f}%  {bar}")
    print("  " + "-" * 58)
    total_correct = df["correct"].sum()
    total = len(df)
    print(f"  {'Overall':<10} {total_correct:>8} "
          f"{total - total_correct:>10} {total:>7} "
          f"  {100*total_correct/total:>5.1f}%")
    print("=" * 62)
    print()

    print("  Per-dataset breakdown:")
    for ds in ["healthcare", "ecommerce", "government"]:
        sub = df[df["dataset"] == ds]
        c = sub["correct"].sum()
        t = len(sub)
        print(f"    {ds:<12}: {c}/{t} correct ({100*c/t:.0f}%)")
    print()

    print("  Per-issue-type breakdown:")
    for it in sorted(df["issue_type"].unique()):
        sub = df[df["issue_type"] == it]
        c = sub["correct"].sum()
        t = len(sub)
        print(f"    {it:<22}: {c}/{t} correct ({100*c/t:.0f}%)")
    print()


def plot_precision(matrix: pd.DataFrame) -> Path:
    """Save a horizontal bar chart of precision-at-tier."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor("#1E1E1E")
    ax.set_facecolor("#1E1E1E")

    tiers = [t for t in TIER_ORDER if t in matrix.index]
    precisions = [matrix.loc[t, "Precision"] for t in tiers]
    totals = [int(matrix.loc[t, "Total"]) for t in tiers]
    colors = [TIER_COLORS[t] for t in tiers]

    bars = ax.barh(tiers, precisions, color=colors, height=0.5, zorder=3)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Precision (%)", color="white", fontsize=11)
    ax.set_title(
        "Precision-at-Tier — Confidence Scoring Calibration",
        color="white", fontsize=12, fontweight="bold", pad=12
    )
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.set_facecolor("#1E1E1E")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.tick_params(axis="x", colors="#AAAAAA")
    ax.tick_params(axis="y", colors="white", labelsize=12)
    ax.grid(axis="x", color="#333", linestyle="--", linewidth=0.6, zorder=0)

    for bar, prec, total in zip(bars, precisions, totals):
        ax.text(
            prec + 1.5, bar.get_y() + bar.get_height() / 2,
            f"{prec:.0f}%  (n={total})",
            va="center", ha="left", color="white", fontsize=11, fontweight="bold"
        )

    plt.tight_layout()
    out = OUT_DIR / "confusion_matrix_precision.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1E1E1E")
    plt.close()
    print(f"  Chart saved to: {out}")
    return out


def main() -> None:
    if not SCORES_CSV.exists():
        print(f"Error: {SCORES_CSV} not found.")
        print("Run scripts/score_proposals.py first.")
        sys.exit(1)

    df = pd.read_csv(SCORES_CSV)
    matrix = build_matrix(df)

    print_matrix(matrix, df)
    plot_precision(matrix)

    # Save the matrix as CSV too
    matrix_path = OUT_DIR / "confusion_matrix.csv"
    matrix.to_csv(matrix_path)
    print(f"  Matrix saved to: {matrix_path}")


if __name__ == "__main__":
    main()
