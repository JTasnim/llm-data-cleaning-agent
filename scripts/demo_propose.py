#!/usr/bin/env python3
"""
Manual demo: run the full Layer 1 + Layer 2 pipeline on a benchmark dataset
and print every cleaning proposal Gemini generates — useful for inspecting
what the agent proposes before the dry-run verification step is built.

Usage:
    python scripts/demo_propose.py
    python scripts/demo_propose.py --dataset ecommerce_superstore_sales.csv
    python scripts/demo_propose.py --corrupted   # use the corrupted version
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.agent.domain_inference import infer_domain
from src.agent.propose import propose_transforms
from src.profiler.profile import profile_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="healthcare_pima_diabetes.csv",
        help="Filename to analyze (default: %(default)s)",
    )
    parser.add_argument(
        "--corrupted", action="store_true",
        help="Use the corrupted version from benchmarks/corrupted/ instead of raw",
    )
    args = parser.parse_args()

    folder = "corrupted" if args.corrupted else "raw"
    csv_path = REPO_ROOT / "benchmarks" / folder / args.dataset
    if not csv_path.exists():
        print(f"Could not find {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Dataset: {args.dataset} ({folder}), {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("Step 1: running Layer 1 profiler ...")
    profile = profile_dataset(df)
    print(f"  found: {profile.duplicate_row_count} duplicate rows")
    for name, col in profile.columns.items():
        if col.null_count > 0 or col.outlier_count > 0:
            print(f"  {name}: {col.null_count} nulls, {col.outlier_count} outliers")

    print("\nStep 2: running domain inference ...")
    domain = infer_domain(df, n_sample_rows=5)
    print(f"  domain: {domain.domain} (confidence: {domain.domain_confidence})")

    print("\nStep 3: generating transform proposals ...")
    proposals = propose_transforms(df, profile, domain, n_sample_rows=5)

    print(f"\n{'='*60}")
    print(f"  {len(proposals)} proposals generated")
    print(f"{'='*60}\n")

    for i, p in enumerate(proposals, 1):
        print(f"Proposal {i} [{p.issue_type}] — column: {p.column}")
        print(f"  Issue:    {p.description}")
        print(f"  Fix:      {p.proposed_fix}")
        print(f"  Code:     {p.transform_code}")
        print(f"  Affected: {p.affected_count} rows")
        print(f"  Confidence tier: {p.confidence_tier}")
        print()


if __name__ == "__main__":
    main()
