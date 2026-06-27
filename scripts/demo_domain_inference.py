#!/usr/bin/env python3
"""
Manual demo: run domain inference on one of the real benchmark datasets and
print what Gemini actually infers — useful for eyeballing output quality
before wiring this into the full agent loop.

This is NOT a test (see tests/test_domain_inference.py for that) — it's a
one-off script for human inspection, requires a real GOOGLE_API_KEY in .env.

Usage:
    python scripts/demo_domain_inference.py
    python scripts/demo_domain_inference.py --dataset ecommerce_superstore_sales.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.agent.domain_inference import infer_domain

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "benchmarks" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="healthcare_pima_diabetes.csv",
        help="Filename in benchmarks/raw/ to analyze (default: %(default)s)",
    )
    args = parser.parse_args()

    csv_path = RAW_DIR / args.dataset
    if not csv_path.exists():
        print(f"Could not find {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Running domain inference on {args.dataset} ({df.shape[0]} rows, {df.shape[1]} columns)...\n")

    result = infer_domain(df)

    print(f"Inferred domain: {result.domain}  (confidence: {result.domain_confidence})")
    print(f"Reasoning: {result.reasoning}\n")
    print("Per-column semantics:")
    for col in result.columns:
        print(f"  {col.column}: {col.inferred_meaning} (plausible: {col.plausible_range_or_values})")


if __name__ == "__main__":
    main()
