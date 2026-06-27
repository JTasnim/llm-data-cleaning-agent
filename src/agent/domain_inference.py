"""
Domain inference — the first reasoning step of Layer 2 (src/agent/).

Given a dataset's column names and a small sample of rows, asks the LLM
backbone (Gemini 2.5 Flash, free tier) to infer what real-world domain the
dataset comes from and what each column semantically represents. This
semantic grounding is a prerequisite for everything downstream: detecting
whether a value is "wrong" requires first knowing what the column is
supposed to mean (e.g. recognizing that BloodPressure=0 is implausible
requires knowing the column is a blood pressure reading, not just that it's
a numeric column with an unusual value).

This module deliberately does ONE thing — domain + column-semantics
inference — so it can be built, tested, and trusted before being wired into
the larger LangGraph agent loop (propose -> dry-run -> score), which still
lives as a stub in src/agent/agent.py.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_SAMPLE_ROWS = 5


@dataclass
class ColumnSemantics:
    column: str
    inferred_meaning: str
    plausible_range_or_values: str


@dataclass
class DomainInferenceResult:
    domain: str
    domain_confidence: str  # "High" | "Medium" | "Low" — the model's own self-rated confidence
    reasoning: str
    columns: list[ColumnSemantics]

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "domain_confidence": self.domain_confidence,
            "reasoning": self.reasoning,
            "columns": [
                {
                    "column": c.column,
                    "inferred_meaning": c.inferred_meaning,
                    "plausible_range_or_values": c.plausible_range_or_values,
                }
                for c in self.columns
            ],
        }


_PROMPT_TEMPLATE = """You are a data analyst inferring the domain and semantics of a tabular dataset.

Column names: {columns}

Sample rows (as JSON):
{sample_rows_json}

Respond with ONLY a JSON object (no markdown fences, no preamble) matching this exact shape:
{{
  "domain": "<one short phrase, e.g. 'healthcare / diabetes screening'>",
  "domain_confidence": "<High, Medium, or Low>",
  "reasoning": "<1-2 sentences explaining why you inferred this domain>",
  "columns": [
    {{
      "column": "<exact column name as given>",
      "inferred_meaning": "<what this column represents in plain English>",
      "plausible_range_or_values": "<a realistic range for numeric columns, or a short list of expected categories for categorical columns>"
    }}
  ]
}}

Include one entry in "columns" for every column name given above, in the same order.
"""


def _build_prompt(columns: list[str], sample_rows: list[dict]) -> str:
    return _PROMPT_TEMPLATE.format(
        columns=", ".join(columns),
        sample_rows_json=json.dumps(sample_rows, indent=2, default=str),
    )


def _parse_response(raw_text: str, expected_columns: list[str]) -> DomainInferenceResult:
    """Parse the model's JSON response, tolerating common formatting quirks
    (e.g. wrapping the JSON in markdown code fences despite instructions).
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    data = json.loads(text)

    columns = [
        ColumnSemantics(
            column=c["column"],
            inferred_meaning=c["inferred_meaning"],
            plausible_range_or_values=c["plausible_range_or_values"],
        )
        for c in data["columns"]
    ]

    returned_names = {c.column for c in columns}
    missing = set(expected_columns) - returned_names
    if missing:
        raise ValueError(
            f"Model response is missing semantics for columns: {sorted(missing)}"
        )

    return DomainInferenceResult(
        domain=data["domain"],
        domain_confidence=data["domain_confidence"],
        reasoning=data["reasoning"],
        columns=columns,
    )


def infer_domain(
    df: pd.DataFrame,
    n_sample_rows: int = DEFAULT_SAMPLE_ROWS,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> DomainInferenceResult:
    """Infer the dataset's domain and per-column semantics using Gemini.

    Args:
        df: the dataframe to analyze (only a small sample is sent to the
            model — never the full dataset).
        n_sample_rows: how many rows to sample and include in the prompt.
        model: the Gemini model name to use.
        api_key: explicit API key; falls back to the GOOGLE_API_KEY env var
            (loaded from .env via python-dotenv) if not provided.

    Returns:
        A DomainInferenceResult with the inferred domain and per-column
        semantics, in the same column order as the input dataframe.

    Raises:
        ValueError: if GOOGLE_API_KEY is not set, or if the model's
            response is missing required fields or columns.
        json.JSONDecodeError: if the model's response isn't valid JSON.
    """
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY not set. Add it to your .env file "
            "(see .env.example) or pass api_key explicitly."
        )

    columns = list(df.columns)
    sample_rows = df.head(n_sample_rows).to_dict(orient="records")

    prompt = _build_prompt(columns, sample_rows)

    client = genai.Client(api_key=key)
    response = client.models.generate_content(model=model, contents=prompt)

    return _parse_response(response.text, expected_columns=columns)
