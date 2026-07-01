"""
Transform proposal — Layer 2 (src/agent/).

Takes the Layer 1 DatasetProfile and the domain-inference result together
and asks Gemini to propose concrete cleaning transformations: one proposal
per detected issue, each with a plain-English explanation and the Python /
Pandas code that would fix it.

This is the second reasoning step in the agent pipeline:
    profile  +  domain semantics  -->  [propose]  -->  CleaningProposals
                                                            |
                                                     (next step: dry-run
                                                      self-verification
                                                      + confidence scoring)

The confidence tier (High / Medium / Low) on each proposal is left as
"Unverified" at this stage — it will be filled in by the dry-run
self-verification step (src/agent/verify.py, Phase 2 Week 4-5), which
executes each proposal's code against a sandboxed copy of the data and
reports the outcome back to the scoring mechanism in src/agent/confidence.py.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import pandas as pd
from dotenv import load_dotenv
from google import genai

from src.agent.domain_inference import DomainInferenceResult
from src.profiler.profile import DatasetProfile

load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"
UNVERIFIED = "Unverified"


@dataclass
class CleaningProposal:
    """A single proposed data-cleaning transformation.

    Attributes:
        issue_type: short category label — one of "missing_value",
            "outlier", "duplicate", "label_inconsistency", "type_mismatch",
            or "domain_implausible" (a value that is statistically normal
            but semantically wrong, e.g. BloodPressure=0).
        column: the column the issue was found in, or "<row>" for
            row-level issues such as duplicates.
        description: plain-English description of the issue, written for
            a human reviewer who will approve or reject this proposal.
        proposed_fix: plain-English description of the recommended fix.
        transform_code: Python / Pandas code that implements the fix. The
            code must assume the dataframe is named `df` and must be
            self-contained (no imports). This code will be executed in a
            sandboxed environment during dry-run verification.
        affected_count: how many rows / values are affected.
        confidence_tier: "Unverified" until the dry-run step fills it in.
            After verification it will be "High", "Medium", or "Low".
    """

    issue_type: str
    column: str
    description: str
    proposed_fix: str
    transform_code: str
    affected_count: int
    confidence_tier: str = UNVERIFIED

    def to_dict(self) -> dict:
        return {
            "issue_type": self.issue_type,
            "column": self.column,
            "description": self.description,
            "proposed_fix": self.proposed_fix,
            "transform_code": self.transform_code,
            "affected_count": self.affected_count,
            "confidence_tier": self.confidence_tier,
        }


_PROMPT_TEMPLATE = """You are an expert data analyst generating data-cleaning proposals.

You have been given:
1. A statistical profile of a dataset (null rates, outliers, duplicates, etc.)
2. Domain knowledge about what each column represents and its plausible values
3. A small sample of actual rows for context

Your job is to propose concrete cleaning transformations for any data-quality
issues you detect. Consider BOTH:
  - Statistical issues (high null rate, numeric outliers, duplicate rows)
  - Semantic issues (values that are statistically normal but domain-implausible,
    e.g. BloodPressure = 0 in a medical dataset, or a Discount > 1 in e-commerce)

Domain: {domain}

Statistical profile:
{profile_json}

Column semantics (what each column means and its plausible range/values):
{semantics_json}

Sample rows:
{sample_rows_json}

Respond with ONLY a JSON array (no markdown fences, no preamble). Each element
must match this exact shape:
{{
  "issue_type": "<missing_value | outlier | duplicate | label_inconsistency | type_mismatch | domain_implausible>",
  "column": "<exact column name, or '<row>' for row-level issues>",
  "description": "<1-2 sentences describing the issue for a human reviewer>",
  "proposed_fix": "<1 sentence describing the recommended fix in plain English>",
  "transform_code": "<self-contained Python/Pandas code; assume dataframe is named df; no imports needed>",
  "affected_count": <integer number of affected rows or values>
}}

Rules:
- Generate ONE proposal per distinct issue. Do not duplicate.
- For missing values: only propose if null_rate > 0.01 (more than 1%).
- For outliers: only propose if the outlier_count > 0.
- For domain_implausible: use the plausible_range_or_values to judge. Be specific
  about which values are implausible and why.
- For duplicates: only propose if duplicate_row_count > 0.
- Do not propose fixes for issues that do not exist in the profile.
- Keep transform_code short and safe. Use df.fillna(), df.drop_duplicates(),
  df[col].replace(), df.loc[mask, col] = value patterns.
- affected_count must be a realistic integer derived from the profile data.
"""


def _build_prompt(
    profile: DatasetProfile,
    domain_result: DomainInferenceResult,
    sample_rows: list[dict],
) -> str:
    profile_summary = {
        "n_rows": profile.n_rows,
        "n_columns": profile.n_columns,
        "duplicate_row_count": profile.duplicate_row_count,
        "columns": {
            name: {
                "dtype": col.dtype,
                "null_count": col.null_count,
                "null_rate": round(col.null_rate, 4),
                "is_numeric": col.is_numeric,
                "is_mixed_type": col.is_mixed_type,
                "outlier_count": col.outlier_count,
                "min": col.min,
                "max": col.max,
                "mean": round(col.mean, 2) if col.mean is not None else None,
            }
            for name, col in profile.columns.items()
        },
    }

    semantics = [
        {
            "column": c.column,
            "inferred_meaning": c.inferred_meaning,
            "plausible_range_or_values": c.plausible_range_or_values,
        }
        for c in domain_result.columns
    ]

    return _PROMPT_TEMPLATE.format(
        domain=domain_result.domain,
        profile_json=json.dumps(profile_summary, indent=2),
        semantics_json=json.dumps(semantics, indent=2),
        sample_rows_json=json.dumps(sample_rows, indent=2, default=str),
    )


def _parse_proposals(raw_text: str) -> list[CleaningProposal]:
    """Parse the model's JSON array response into CleaningProposal objects."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    items = json.loads(text)
    if not isinstance(items, list):
        raise ValueError(f"Expected a JSON array, got {type(items).__name__}")

    proposals = []
    for item in items:
        proposals.append(
            CleaningProposal(
                issue_type=item["issue_type"],
                column=item["column"],
                description=item["description"],
                proposed_fix=item["proposed_fix"],
                transform_code=item["transform_code"],
                affected_count=int(item["affected_count"]),
                confidence_tier=UNVERIFIED,
            )
        )
    return proposals


def propose_transforms(
    df: pd.DataFrame,
    profile: DatasetProfile,
    domain_result: DomainInferenceResult,
    n_sample_rows: int = 5,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> list[CleaningProposal]:
    """Generate cleaning proposals by combining the statistical profile
    with domain semantics via Gemini.

    Args:
        df: the dataframe being cleaned (used only for sample rows —
            never modified).
        profile: Layer 1 DatasetProfile from src/profiler/profile.py.
        domain_result: domain-inference result from
            src/agent/domain_inference.py.
        n_sample_rows: number of rows to include in the prompt.
        model: Gemini model name.
        api_key: explicit key; falls back to GOOGLE_API_KEY env var.

    Returns:
        A list of CleaningProposal objects, each with confidence_tier
        set to "Unverified" pending dry-run verification.
    """
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY not set. Add it to your .env file or pass api_key explicitly."
        )

    sample_rows = df.head(n_sample_rows).to_dict(orient="records")
    prompt = _build_prompt(profile, domain_result, sample_rows)

    client = genai.Client(api_key=key)
    response = client.models.generate_content(model=model, contents=prompt)

    return _parse_proposals(response.text)
