#!/usr/bin/env python3
"""
Build the evaluation benchmark: read each clean dataset in benchmarks/raw/,
apply the synthetic error-injection pipeline with dataset-specific
parameters, write the corrupted CSV + ground-truth ledger, then run the
Layer 1 profiler over the corrupted file and print a short summary so you
can sanity-check that the profiler actually re-discovers what was injected.

Usage:
    python scripts/build_benchmark.py
    python scripts/build_benchmark.py --seed 7      # different corruption run
    python scripts/build_benchmark.py --dataset healthcare_pima_diabetes.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.evaluation.error_injection import corrupt_dataset
from src.profiler.profile import profile_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "benchmarks" / "raw"
CORRUPTED_DIR = REPO_ROOT / "benchmarks" / "corrupted"
GROUND_TRUTH_DIR = REPO_ROOT / "benchmarks" / "ground_truth"

# Per-dataset corruption parameters. Each tuned roughly to dataset size:
# ~5% missing rate everywhere; outlier/duplicate counts scale with row count
# so smaller datasets aren't disproportionately flooded with injected noise.
DATASET_CONFIGS = {
    "healthcare_pima_diabetes.csv": dict(
        missing_columns=["BMI", "Glucose"],
        missing_rate=0.05,
        outlier_column="Age",
        n_outliers=8,
        n_duplicates=8,
    ),
    "ecommerce_superstore_sales.csv": dict(
        missing_columns=["Sales", "Discount"],
        missing_rate=0.05,
        outlier_column="Profit",
        n_outliers=20,
        n_duplicates=20,
    ),
    "government_adult_income.csv": dict(
        missing_columns=["age", "hours_per_week"],
        missing_rate=0.05,
        label_column="native_country",
        label_variant_map={
            "United-States": ["USA", "U.S.A.", "US", "United States"],
        },
        outlier_column="capital_gain",
        n_outliers=50,
        n_duplicates=50,
    ),
}


def build_one(filename: str, seed: int) -> None:
    raw_path = RAW_DIR / filename
    if not raw_path.exists():
        print(f"  [skip] {filename} not found in {RAW_DIR}")
        return

    config = DATASET_CONFIGS.get(filename, {})
    if not config:
        print(f"  [skip] no corruption config defined for {filename}")
        return

    print(f"\n=== {filename} ===")
    df = pd.read_csv(raw_path)
    print(f"  raw shape: {df.shape}")

    result = corrupt_dataset(df, seed=seed, **config)
    print(f"  corrupted shape: {result.corrupted_df.shape}  "
          f"({len(result.ledger)} ground-truth ledger entries)")

    CORRUPTED_DIR.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    corrupted_path = CORRUPTED_DIR / filename
    result.corrupted_df.to_csv(corrupted_path, index=False)

    ledger_path = GROUND_TRUTH_DIR / filename.replace(".csv", "_ledger.csv")
    result.ledger_as_dataframe().to_csv(ledger_path, index=False)

    print(f"  wrote: {corrupted_path.relative_to(REPO_ROOT)}")
    print(f"  wrote: {ledger_path.relative_to(REPO_ROOT)}")

    # Sanity-check: re-profile the corrupted file and report what the
    # profiler independently rediscovers, so you can eyeball that the
    # numbers line up with what was actually injected (Layer 1 <-> Layer 2
    # consistency check, ahead of building the agent in Phase 2).
    profile = profile_dataset(result.corrupted_df)
    print(f"  profiler found: {profile.duplicate_row_count} duplicate rows; "
          f"nulls -> " + ", ".join(
              f"{name}={col.null_count}" for name, col in profile.columns.items()
              if col.null_count > 0
          ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible corruption (default: 42)")
    parser.add_argument("--dataset", type=str, default=None, help="Build only this one filename instead of all three")
    args = parser.parse_args()

    targets = [args.dataset] if args.dataset else list(DATASET_CONFIGS.keys())

    print(f"Building benchmark with seed={args.seed} ...")
    for filename in targets:
        build_one(filename, seed=args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
