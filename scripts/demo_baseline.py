#!/usr/bin/env python3
"""
Naive LLM baseline (control condition) — Phase 4 evaluation.

This is the control condition for the experimental design. It calls Gemini
with only the raw statistical profile — no domain inference, no dry-run
self-verification. All proposals come back as "Unverified".

This directly tests H2: a system WITH dry-run verification should produce
fewer broken proposals than one without it.

Compare results against scripts/score_proposals.py output (full system).

Usage:
    python scripts/demo_baseline.py
    python scripts/demo_baseline.py --dataset ecommerce_superstore_sales.csv
    python scripts/demo_baseline.py --corrupted
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from dotenv import load_dotenv
from google import genai

from src.profiler.profile import profile_dataset

load_dotenv()

REPO_ROOT  = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gemini-2.5-flash"

# ── Baseline proposal dataclass ───────────────────────────────────────────────

@dataclass
class BaselineProposal:
    issue_type:     str
    column:         str
    description:    str
    proposed_fix:   str
    transform_code: str
    affected_count: int
    confidence_tier: str = "Unverified"  # always Unverified — no dry-run

    def to_dict(self) -> dict:
        return {
            "issue_type":      self.issue_type,
            "column":          self.column,
            "description":     self.description,
            "proposed_fix":    self.proposed_fix,
            "transform_code":  self.transform_code,
            "affected_count":  self.affected_count,
            "confidence_tier": self.confidence_tier,
        }


# ── Baseline prompt ───────────────────────────────────────────────────────────
# Deliberately simpler than the full system prompt:
#   - No domain semantics (no infer_domain step)
#   - No zero_count field
#   - No CoW-safe instruction
#   - No dry-run verification after this call
# This represents what a naive LLM-only approach produces.

_BASELINE_PROMPT = """You are a data analyst generating data-cleaning proposals.

You have been given a statistical profile of a dataset and a small sample
of rows. Propose concrete cleaning transformations for any data-quality
issues you detect (missing values, outliers, duplicates, type mismatches).

Statistical profile:
{profile_json}

Sample rows:
{sample_rows_json}

Respond with ONLY a JSON array (no markdown fences, no preamble). Each
element must match this exact shape:
{{
  "issue_type": "<missing_value | outlier | duplicate | label_inconsistency | type_mismatch>",
  "column": "<exact column name, or '<row>' for row-level issues>",
  "description": "<1-2 sentences describing the issue>",
  "proposed_fix": "<recommended fix in plain English>",
  "transform_code": "<self-contained Python/Pandas code; df is the dataframe>",
  "affected_count": <integer>
}}

Rules:
- ONE proposal per distinct issue.
- For missing values: only propose if null_rate > 0.01.
- For outliers: only propose if outlier_count > 0.
- For duplicates: only propose if duplicate_row_count > 0.
- Do NOT propose fixes for issues that do not exist in the profile.
"""


def _call_gemini(prompt: str, model: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model, contents=prompt
            )
            return response.text
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                wait = 10 * (attempt + 1)
                print(f"  Gemini unavailable, retrying in {wait}s ...")
                time.sleep(wait)
                if attempt == 2:
                    raise
            else:
                raise


def _parse_proposals(raw_text: str) -> list[BaselineProposal]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    items = json.loads(text)
    if not isinstance(items, list):
        raise ValueError(f"Expected JSON array, got {type(items).__name__}")
    return [
        BaselineProposal(
            issue_type=item["issue_type"],
            column=item["column"],
            description=item["description"],
            proposed_fix=item["proposed_fix"],
            transform_code=item["transform_code"],
            affected_count=int(item["affected_count"]),
            confidence_tier="Unverified",
        )
        for item in items
    ]


def run_baseline(df: pd.DataFrame, n_sample: int = 5,
                 model: str = DEFAULT_MODEL) -> list[BaselineProposal]:
    """Run the naive baseline: profile → Gemini proposals (no verification)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set.")

    # Layer 1: profiler (same as full system)
    profile = profile_dataset(df)

    # Build a minimal profile summary — no zero_count, no domain semantics
    profile_summary = {
        "n_rows": profile.n_rows,
        "n_columns": profile.n_columns,
        "duplicate_row_count": profile.duplicate_row_count,
        "columns": {
            name: {
                "dtype":        col.dtype,
                "null_count":   col.null_count,
                "null_rate":    round(col.null_rate, 4),
                "is_numeric":   col.is_numeric,
                "outlier_count": col.outlier_count,
                "min":          col.min,
                "max":          col.max,
                "mean":         round(col.mean, 2) if col.mean is not None else None,
            }
            for name, col in profile.columns.items()
        }
    }

    sample_rows = df.head(n_sample).to_dict(orient="records")
    prompt = _BASELINE_PROMPT.format(
        profile_json=json.dumps(profile_summary, indent=2),
        sample_rows_json=json.dumps(sample_rows, indent=2, default=str),
    )

    raw = _call_gemini(prompt, model, api_key)
    return _parse_proposals(raw)


def print_proposals(proposals: list[BaselineProposal],
                    dataset_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  BASELINE RESULTS — {dataset_name}")
    print(f"  {len(proposals)} proposals (all Unverified — no dry-run)")
    print(f"{'='*60}")
    for i, p in enumerate(proposals, 1):
        print(f"\n  Proposal {i} [Unverified] [{p.issue_type}] "
              f"column: {p.column}")
        print(f"  Issue:    {p.description}")
        print(f"  Fix:      {p.proposed_fix}")
        print(f"  Code:     {p.transform_code[:80]}{'...' if len(p.transform_code)>80 else ''}")
        print(f"  Affected: {p.affected_count} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="healthcare_pima_diabetes.csv",
        help="Filename in benchmarks/raw/ or benchmarks/corrupted/"
    )
    parser.add_argument(
        "--corrupted", action="store_true",
        help="Use the corrupted version of the dataset"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all 3 benchmark datasets"
    )
    args = parser.parse_args()

    folder = "corrupted" if args.corrupted else "raw"
    datasets = [
        "healthcare_pima_diabetes.csv",
        "ecommerce_superstore_sales.csv",
        "government_adult_income.csv",
    ] if args.all else [args.dataset]

    all_results = []
    for filename in datasets:
        csv_path = REPO_ROOT / "benchmarks" / folder / filename
        if not csv_path.exists():
            print(f"  [skip] {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        dataset_key = filename.replace(".csv", "").replace(
            "healthcare_pima_diabetes", "healthcare"
        ).replace(
            "ecommerce_superstore_sales", "ecommerce"
        ).replace(
            "government_adult_income", "government"
        )

        print(f"\nRunning baseline on {filename} "
              f"({df.shape[0]} rows, {df.shape[1]} cols) ...")
        proposals = run_baseline(df)
        print_proposals(proposals, filename)

        for i, p in enumerate(proposals, 1):
            all_results.append({
                "dataset":        dataset_key,
                "proposal_num":   i,
                "issue_type":     p.issue_type,
                "column":         p.column,
                "description":    p.description,
                "transform_code": p.transform_code,
                "affected_count": p.affected_count,
                "confidence_tier": "Unverified",
            })

    if all_results:
        out_path = REPO_ROOT / "outputs" / "baseline_proposals.csv"
        pd.DataFrame(all_results).to_csv(out_path, index=False)
        print(f"\n  Saved to: {out_path}")
        print(f"  Total proposals: {len(all_results)} (all Unverified)")
        print("\n  Compare against full system results in:")
        print("  outputs/proposal_scores.csv")


if __name__ == "__main__":
    main()
