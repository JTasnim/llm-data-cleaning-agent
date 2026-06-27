"""Tests for domain inference (src/agent/domain_inference.py).

Two kinds of tests here:
  1. Unit tests for prompt-building and response-parsing logic — these run
     with no API key and no network access.
  2. One live integration test that actually calls Gemini — automatically
     skipped if GOOGLE_API_KEY isn't set, so the suite still passes in
     environments without a key (e.g. CI, or a fresh clone).

Run with: pytest tests/test_domain_inference.py -v
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.domain_inference import (
    ColumnSemantics,
    DomainInferenceResult,
    _build_prompt,
    _parse_response,
    infer_domain,
)


def test_build_prompt_includes_all_column_names():
    prompt = _build_prompt(["age", "income"], [{"age": 30, "income": 50000}])
    assert "age" in prompt
    assert "income" in prompt


def test_build_prompt_embeds_sample_rows_as_json():
    rows = [{"age": 30, "income": 50000}]
    prompt = _build_prompt(["age", "income"], rows)
    assert "50000" in prompt


def _make_valid_response_json(columns: list[str]) -> str:
    return json.dumps(
        {
            "domain": "healthcare / diabetes screening",
            "domain_confidence": "High",
            "reasoning": "Columns match the well-known Pima diabetes dataset.",
            "columns": [
                {
                    "column": col,
                    "inferred_meaning": f"meaning of {col}",
                    "plausible_range_or_values": "0-100",
                }
                for col in columns
            ],
        }
    )


def test_parse_response_handles_clean_json():
    raw = _make_valid_response_json(["Glucose", "BMI"])
    result = _parse_response(raw, expected_columns=["Glucose", "BMI"])
    assert isinstance(result, DomainInferenceResult)
    assert result.domain == "healthcare / diabetes screening"
    assert len(result.columns) == 2
    assert all(isinstance(c, ColumnSemantics) for c in result.columns)


def test_parse_response_strips_markdown_code_fences():
    raw_json = _make_valid_response_json(["Glucose"])
    fenced = f"```json\n{raw_json}\n```"
    result = _parse_response(fenced, expected_columns=["Glucose"])
    assert result.domain == "healthcare / diabetes screening"


def test_parse_response_raises_on_missing_column():
    raw = _make_valid_response_json(["Glucose"])  # only one column
    with pytest.raises(ValueError, match="missing semantics"):
        _parse_response(raw, expected_columns=["Glucose", "BMI"])


def test_parse_response_raises_on_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        _parse_response("this is not json", expected_columns=["Glucose"])


def test_domain_inference_result_to_dict_roundtrips_shape():
    raw = _make_valid_response_json(["Glucose"])
    result = _parse_response(raw, expected_columns=["Glucose"])
    d = result.to_dict()
    assert d["domain"] == "healthcare / diabetes screening"
    assert d["columns"][0]["column"] == "Glucose"


def test_infer_domain_raises_clear_error_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        infer_domain(df, api_key=None)


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — skipping live Gemini integration test",
)
def test_infer_domain_live_call_on_pima_like_sample():
    """Live test: actually calls Gemini. Only runs if a real key is present."""
    df = pd.DataFrame(
        {
            "Pregnancies": [6, 1, 8],
            "Glucose": [148, 85, 183],
            "BMI": [33.6, 26.6, 23.3],
            "Age": [50, 31, 32],
            "Outcome": [1, 0, 1],
        }
    )
    result = infer_domain(df, n_sample_rows=3)
    assert isinstance(result.domain, str) and len(result.domain) > 0
    assert result.domain_confidence in {"High", "Medium", "Low"}
    assert len(result.columns) == len(df.columns)
    returned_names = {c.column for c in result.columns}
    assert returned_names == set(df.columns)
