"""
Phase 4 — Confidence-tier confusion matrix + reliability diagram.

Reads proposal_scores.csv produced by scripts/score_proposals.py and:
  1. Prints the formal confusion matrix table
  2. Computes precision-at-tier for High / Medium / Low
  3. Saves a precision-at-tier bar chart
  4. Saves a reliability diagram (calibration plot)
     - x-axis: mean predicted confidence score per tier
     - y-axis: actual accuracy (fraction correct) per tier
     - diagonal = perfect calibration
     This directly addresses the discrimination vs. calibration distinction:
     the confusion matrix shows discrimination; the reliability diagram
     shows whether the confidence scores are actually calibrated.

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
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORES_CSV = REPO_ROOT / "outputs" / "proposal_scores.csv"
OUT_DIR    = REPO_ROOT / "outputs"
FIGS_DIR   = REPO_ROOT / "docs" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

TIER_ORDER  = ["High", "Medium", "Low"]
TIER_COLORS = {"High": "#28A745", "Medium": "#FFC107", "Low": "#DC3545"}

# Mean confidence score per tier (from verify.py scoring)
TIER_MEAN_SCORES = {"High": 0.95, "Medium": 0.75, "Low": 0.05}


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate predicted tier vs. actual correctness.
    Only uses scoreable proposals (label == 'correct' or 'incorrect').
    """
    scoreable = df[df["label"] != "natural_correct"].copy() \
        if "label" in df.columns else df.copy()

    matrix = pd.crosstab(
        scoreable["predicted_tier"],
        scoreable["correct"],
        rownames=["Predicted tier"],
        colnames=["Actually correct"]
    ).reindex(TIER_ORDER, fill_value=0)

    matrix.columns = [str(c) for c in matrix.columns]
    if "True"  not in matrix.columns: matrix["True"]  = 0
    if "False" not in matrix.columns: matrix["False"] = 0
    matrix = matrix[["True", "False"]]
    matrix.columns = ["Correct", "Incorrect"]
    matrix["Total"]     = matrix["Correct"] + matrix["Incorrect"]
    matrix["Precision"] = (
        matrix["Correct"] / matrix["Total"].replace(0, float("nan")) * 100
    ).round(1)
    return matrix


def print_matrix(matrix: pd.DataFrame, df: pd.DataFrame) -> None:
    scoreable = df[df["label"] != "natural_correct"] \
        if "label" in df.columns else df

    print()
    print("=" * 65)
    print("  Confidence-Tier Confusion Matrix (strict scoring)")
    print("  Correctness = column match only, rows_affected not used")
    print("=" * 65)
    print(f"  {'Tier':<10} {'Correct':>8} {'Incorrect':>10} "
          f"{'Total':>7} {'Precision':>10}")
    print("  " + "-" * 60)

    for tier in TIER_ORDER:
        if tier not in matrix.index or matrix.loc[tier, "Total"] == 0:
            continue
        row = matrix.loc[tier]
        bar = "█" * int(row["Precision"] / 10)
        print(f"  {tier:<10} {int(row['Correct']):>8} "
              f"{int(row['Incorrect']):>10} {int(row['Total']):>7} "
              f"  {row['Precision']:>5.1f}%  {bar}")

    print("  " + "-" * 60)
    tc = scoreable["correct"].sum()
    t  = len(scoreable)
    print(f"  {'Overall':<10} {int(tc):>8} {int(t-tc):>10} {t:>7} "
          f"  {100*tc/t:>5.1f}%")
    print("=" * 65)
    print()

    if "label" in df.columns:
        natural = df[df["label"] == "natural_correct"]
        print(f"  Scoreable proposals: {len(scoreable)}")
        print(f"  Natural issues (excluded from matrix): {len(natural)}")
        print(f"  Columns: {sorted(natural['column'].unique())}")
        print()

    print("  Per-dataset breakdown (scoreable only):")
    for ds in ["healthcare", "ecommerce", "government"]:
        sub = scoreable[scoreable["dataset"] == ds]
        if len(sub) == 0:
            continue
        c = sub["correct"].sum()
        print(f"    {ds:<14}: {int(c)}/{len(sub)} correct "
              f"({100*c/len(sub):.0f}%)")
    print()


def plot_precision(matrix: pd.DataFrame) -> None:
    """Save precision-at-tier bar chart."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor("#1E1E1E")
    ax.set_facecolor("#1E1E1E")

    tiers = [t for t in TIER_ORDER
             if t in matrix.index and matrix.loc[t, "Total"] > 0]
    precisions = [matrix.loc[t, "Precision"] for t in tiers]
    totals     = [int(matrix.loc[t, "Total"]) for t in tiers]
    colors     = [TIER_COLORS[t] for t in tiers]

    bars = ax.barh(tiers, precisions, color=colors, height=0.5, zorder=3)
    ax.set_xlim(0, 120)
    ax.set_xlabel("Precision (%)", color="white", fontsize=11)
    ax.set_title("Precision-at-Tier — Confidence Scoring",
                 color="white", fontsize=12, fontweight="bold", pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.tick_params(axis="x", colors="#AAAAAA")
    ax.tick_params(axis="y", colors="white", labelsize=12)
    ax.grid(axis="x", color="#333", linestyle="--", linewidth=0.6, zorder=0)

    for bar, prec, total in zip(bars, precisions, totals):
        ax.text(prec + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{prec:.0f}%  (n={total})",
                va="center", ha="left", color="white",
                fontsize=11, fontweight="bold")

    plt.tight_layout()
    for d in [OUT_DIR, FIGS_DIR]:
        plt.savefig(d / "confusion_matrix_precision.png",
                    dpi=150, bbox_inches="tight", facecolor="#1E1E1E")
    plt.close()
    print(f"  Precision chart saved.")


def plot_reliability_diagram(matrix: pd.DataFrame) -> None:
    """
    Reliability diagram (calibration plot).

    x-axis: mean predicted confidence score per tier
    y-axis: actual fraction correct per tier
    diagonal: perfect calibration line

    A well-calibrated system has points near the diagonal — meaning
    when it says 'I am 95% confident', it is actually correct ~95%
    of the time.

    This is distinct from the confusion matrix (which shows discrimination
    — whether High outperforms Low). The reliability diagram shows
    calibration — whether the confidence numbers themselves are meaningful.

    Note: with a small number of proposals, each point carries wide
    uncertainty. Error bars show 95% Wilson confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#1E1E1E")
    ax.set_facecolor("#1E1E1E")

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--",
            linewidth=1.5, label="Perfect calibration", zorder=1)

    # Shade the overconfidence / underconfidence regions
    ax.fill_between([0, 1], [0, 1], [0, 0],
                    alpha=0.06, color="#DC3545", label="Overconfident region")
    ax.fill_between([0, 1], [0, 1], [1, 1],
                    alpha=0.06, color="#28A745", label="Underconfident region")

    plotted = []
    for tier in TIER_ORDER:
        if tier not in matrix.index or matrix.loc[tier, "Total"] == 0:
            continue
        n       = int(matrix.loc[tier, "Total"])
        correct = int(matrix.loc[tier, "Correct"])
        acc     = correct / n
        x       = TIER_MEAN_SCORES[tier]

        # Wilson 95% confidence interval for the accuracy estimate
        z = 1.96
        denom = 1 + z**2 / n
        centre = (acc + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(acc * (1 - acc) / n + z**2 / (4 * n**2)) / denom
        ci_lo  = max(0.0, centre - margin)
        ci_hi  = min(1.0, centre + margin)

        color = TIER_COLORS[tier]
        ax.errorbar(x, acc,
                    yerr=[[acc - ci_lo], [ci_hi - acc]],
                    fmt="o", color=color, markersize=12,
                    capsize=6, capthick=2, elinewidth=2,
                    zorder=3, label=f"{tier} (n={n})")

        ax.annotate(
            f"{tier}\n{acc:.0%}",
            xy=(x, acc),
            xytext=(x + 0.04, acc - 0.08),
            color=color, fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, lw=1)
        )
        plotted.append((x, acc))

    ax.set_xlim(-0.05, 1.10)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel("Mean predicted confidence score", color="white", fontsize=12)
    ax.set_ylabel("Actual accuracy (fraction correct)", color="white", fontsize=12)
    ax.set_title("Reliability Diagram — Calibration Check\n"
                 "Points near diagonal = well-calibrated",
                 color="white", fontsize=12, fontweight="bold", pad=10)

    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.tick_params(colors="#AAAAAA", labelsize=10)
    ax.grid(color="#333", linestyle="--", linewidth=0.5, zorder=0)

    legend = ax.legend(loc="upper left", fontsize=9,
                       facecolor="#2A2A2A", edgecolor="#555",
                       labelcolor="white")

    # Small-sample caveat note
    ax.text(0.5, 0.04,
            "Note: wide CIs reflect small sample size (n=17 scoreable proposals)",
            ha="center", va="bottom", color="#888888", fontsize=9,
            transform=ax.transAxes)

    plt.tight_layout()
    for d in [OUT_DIR, FIGS_DIR]:
        plt.savefig(d / "reliability_diagram.png",
                    dpi=150, bbox_inches="tight", facecolor="#1E1E1E")
    plt.close()
    print(f"  Reliability diagram saved.")


def main() -> None:
    if not SCORES_CSV.exists():
        print(f"Error: {SCORES_CSV} not found.")
        print("Run scripts/score_proposals.py first.")
        sys.exit(1)

    df = pd.read_csv(SCORES_CSV)

    # Add label column if running against old scores CSV format
    if "label" not in df.columns:
        df["label"] = df["correct"].map(
            {True: "correct", False: "incorrect"}
        )

    matrix = build_matrix(df)
    print_matrix(matrix, df)
    plot_precision(matrix)
    plot_reliability_diagram(matrix)

    matrix_path = OUT_DIR / "confusion_matrix.csv"
    matrix.to_csv(matrix_path)
    print(f"\n  Matrix saved to: {matrix_path}")


if __name__ == "__main__":
    main()
