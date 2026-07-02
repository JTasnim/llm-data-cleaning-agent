#!/usr/bin/env python3
"""
End-to-end pipeline demo: runs the full Phase 2 agent pipeline on a
benchmark dataset and prints each proposal with its verified confidence
tier — the first time all four pieces run together as one system.

Pipeline:
    Raw CSV
        -> Layer 1: statistical profiler
        -> Layer 2a: domain inference (Gemini)
        -> Layer 2b: transform proposals (Gemini)
        -> Layer 2c: dry-run self-verification + confidence scoring
        -> Proposal cards printed for human review (Layer 3 preview)

Usage:
    python scripts/demo_pipeline.py
    python scripts/demo_pipeline.py --dataset ecommerce_superstore_sales.csv
    python scripts/demo_pipeline.py --corrupted   # use corrupted version
    python scripts/demo_pipeline.py --corrupted --dataset healthcare_pima_diabetes.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.agent.domain_inference import infer_domain
from src.agent.propose import propose_transforms
from src.agent.verify import verify_all_proposals
from src.profiler.profile import profile_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]

TIER_SYMBOL = {"High": "✅", "Medium": "⚠️ ", "Low": "❌", "Unverified": "⏳"}
TIER_LABEL  = {"High": "HIGH", "Medium": "MEDIUM", "Low": "LOW", "Unverified": "UNVERIFIED"}


def print_header(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_step(n: int, text: str) -> None:
    print(f"\nStep {n}: {text} ...")


def print_proposal_card(i: int, proposal, result) -> None:
    tier = proposal.confidence_tier
    symbol = TIER_SYMBOL.get(tier, "?")
    print(f"\n{'─'*60}")
    print(f"  Proposal {i}  {symbol} {TIER_LABEL.get(tier, tier)}"
          f"  (score: {result.score:.2f})")
    print(f"{'─'*60}")
    print(f"  Type:     [{proposal.issue_type}]  Column: {proposal.column}")
    print(f"  Issue:    {proposal.description}")
    print(f"  Fix:      {proposal.proposed_fix}")
    print(f"  Code:     {proposal.transform_code}")
    print(f"  Affected: {proposal.affected_count} rows")
    print()
    print(f"  Dry-run result:")
    print(f"    Executed successfully : {result.executed_successfully}")
    print(f"    Rows actually changed : {result.rows_affected}")
    if result.unexpected_columns_changed:
        print(f"    ⚠️  Side effects on   : {result.unexpected_columns_changed}")
    if result.nulls_introduced:
        print(f"    ⚠️  New nulls introduced: {result.nulls_introduced}")
    if result.error_message:
        print(f"    ❌ Error: {result.error_message[:120]}")
    if result.before_sample and result.after_sample:
        print(f"    Before sample: {result.before_sample}")
        print(f"    After sample : {result.after_sample}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="healthcare_pima_diabetes.csv",
        help="Filename in benchmarks/raw/ or benchmarks/corrupted/ (default: %(default)s)"
    )
    parser.add_argument(
        "--corrupted", action="store_true",
        help="Use the synthetically corrupted version of the dataset"
    )
    args = parser.parse_args()

    folder = "corrupted" if args.corrupted else "raw"
    csv_path = REPO_ROOT / "benchmarks" / folder / args.dataset
    if not csv_path.exists():
        print(f"Error: could not find {csv_path}")
        sys.exit(1)

    print_header(f"LLM Data Cleaning Agent — End-to-End Pipeline")
    print(f"  Dataset : {args.dataset} ({folder})")

    df = pd.read_csv(csv_path)
    print(f"  Shape   : {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Step 1: Statistical profiler ──────────────────────────────────────
    print_step(1, "Running Layer 1 statistical profiler")
    t0 = time.time()
    profile = profile_dataset(df)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Duplicate rows   : {profile.duplicate_row_count}")
    for name, col in profile.columns.items():
        flags = []
        if col.null_count > 0:
            flags.append(f"{col.null_count} nulls")
        if col.outlier_count > 0:
            flags.append(f"{col.outlier_count} outliers")
        if col.is_mixed_type:
            flags.append("mixed type")
        if flags:
            print(f"  {name}: {', '.join(flags)}")

    # ── Step 2: Domain inference ───────────────────────────────────────────
    print_step(2, "Running domain inference (Gemini)")
    t0 = time.time()
    domain = infer_domain(df, n_sample_rows=5)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Domain     : {domain.domain} (confidence: {domain.domain_confidence})")
    print(f"  Reasoning  : {domain.reasoning}")

    # ── Step 3: Transform proposals ───────────────────────────────────────
    print_step(3, "Generating transform proposals (Gemini)")
    t0 = time.time()
    proposals = propose_transforms(df, profile, domain, n_sample_rows=5)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  {len(proposals)} proposals generated (all Unverified)")

    # ── Step 4: Dry-run self-verification ─────────────────────────────────
    print_step(4, "Running dry-run self-verification + confidence scoring")
    t0 = time.time()
    verified = verify_all_proposals(df, proposals)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    tier_counts = {"High": 0, "Medium": 0, "Low": 0}
    for proposal, _ in verified:
        tier_counts[proposal.confidence_tier] = \
            tier_counts.get(proposal.confidence_tier, 0) + 1

    print_header("Results — Proposal Cards")
    print(f"  Total proposals : {len(verified)}")
    print(f"  ✅ High         : {tier_counts['High']}")
    print(f"  ⚠️  Medium       : {tier_counts['Medium']}")
    print(f"  ❌ Low          : {tier_counts['Low']}")

    for i, (proposal, result) in enumerate(verified, 1):
        print_proposal_card(i, proposal, result)

    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"  Next step: human approves / edits / rejects each card")
    print(f"  (Layer 3 — Streamlit HITL UI, Phase 3)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
