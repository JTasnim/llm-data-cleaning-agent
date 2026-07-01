"""Tests for the transform proposal module (src/agent/propose.py).

Unit tests cover proposal parsing and the CleaningProposal dataclass.
The live integration test actually calls Gemini — auto-skipped without a key.

Run with: pytest tests/test_propose.py -v
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.domain_inference import ColumnSemantics, DomainInferenceResult
from src.agent.propose import (
    UNVERIFIED,
    CleaningProposal,
    _build_prompt,
    _parse_proposals,
    propose_transforms,
)
from src.profiler.profile import profile_dataset


def make_domain_result() -> DomainInferenceResult:
    return DomainInferenceResult(
        domain="healthcare / diabetes screening",
        domain_confidence="High",
        reasoning="Medical dataset.",
        columns=[
            ColumnSemantics("Glucose", "Plasma glucose", "40-200 mg/dL"),
            ColumnSemantics("BloodPressure", "Diastolic BP", "30-120 mmHg"),
            ColumnSemantics("BMI", "Body Mass Index", "15-60"),
            ColumnSemantics("Age", "Age in years", "20-85"),
        ],
    )


def make_sample_proposals_json(n: int = 2) -> str:
    items = [
        {
            "issue_type": "missing_value",
            "column": "BMI",
            "description": "BMI has 38 missing values (5% null rate).",
            "proposed_fix": "Impute with median BMI value.",
            "transform_code": "df['BMI'].fillna(df['BMI'].median(), inplace=True)",
            "affected_count": 38,
        },
        {
            "issue_type": "domain_implausible",
            "column": "BloodPressure",
            "description": "35 rows have BloodPressure=0, which is clinically impossible.",
            "proposed_fix": "Replace 0 with NaN and impute with median.",
            "transform_code": "df['BloodPressure'].replace(0, pd.NA, inplace=True); df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace=True)",
            "affected_count": 35,
        },
    ]
    return json.dumps(items[:n])


def test_parse_proposals_returns_list_of_cleaning_proposals():
    raw = make_sample_proposals_json(2)
    proposals = _parse_proposals(raw)
    assert len(proposals) == 2
    assert all(isinstance(p, CleaningProposal) for p in proposals)


def test_parse_proposals_confidence_tier_is_unverified():
    raw = make_sample_proposals_json(1)
    proposals = _parse_proposals(raw)
    assert proposals[0].confidence_tier == UNVERIFIED


def test_parse_proposals_strips_markdown_fences():
    raw = make_sample_proposals_json(1)
    fenced = f"```json\n{raw}\n```"
    proposals = _parse_proposals(fenced)
    assert len(proposals) == 1


def test_parse_proposals_raises_on_invalid_json():
    with pytest.raises(Exception):
        _parse_proposals("not valid json")


def test_parse_proposals_raises_on_non_array_response():
    with pytest.raises(ValueError, match="Expected a JSON array"):
        _parse_proposals(json.dumps({"issue_type": "missing_value"}))


def test_cleaning_proposal_to_dict_has_all_fields():
    proposal = CleaningProposal(
        issue_type="missing_value",
        column="BMI",
        description="BMI has nulls.",
        proposed_fix="Impute with median.",
        transform_code="df['BMI'].fillna(df['BMI'].median(), inplace=True)",
        affected_count=38,
    )
    d = proposal.to_dict()
    assert set(d.keys()) == {
        "issue_type", "column", "description",
        "proposed_fix", "transform_code", "affected_count", "confidence_tier",
    }
    assert d["confidence_tier"] == UNVERIFIED


def test_build_prompt_includes_domain_and_column_names():
    df = pd.DataFrame({"Glucose": [148, 85], "BMI": [33.6, 26.6], "BloodPressure": [72, 66], "Age": [50, 31]})
    profile = profile_dataset(df)
    domain = make_domain_result()
    prompt = _build_prompt(profile, domain, df.head(2).to_dict(orient="records"))
    assert "healthcare" in prompt
    assert "BloodPressure" in prompt
    assert "30-120 mmHg" in prompt


def test_propose_transforms_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    df = pd.DataFrame({"Glucose": [148, 85], "BMI": [33.6, None], "BloodPressure": [0, 66], "Age": [50, 31]})
    profile = profile_dataset(df)
    domain = make_domain_result()
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        propose_transforms(df, profile, domain, api_key=None)


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — skipping live Gemini integration test",
)
def test_propose_transforms_live_on_corrupted_pima():
    """Live test — actually calls Gemini. Only runs with a real API key."""
    df = pd.read_csv(
        Path(__file__).resolve().parents[1]
        / "benchmarks" / "corrupted" / "healthcare_pima_diabetes.csv"
    )
    profile = profile_dataset(df)

    domain = DomainInferenceResult(
        domain="healthcare / diabetes screening",
        domain_confidence="High",
        reasoning="Pima Indians diabetes dataset.",
        columns=[
            ColumnSemantics("Pregnancies", "Number of pregnancies", "0-17"),
            ColumnSemantics("Glucose", "Plasma glucose concentration", "40-200 mg/dL"),
            ColumnSemantics("BloodPressure", "Diastolic blood pressure", "30-120 mmHg"),
            ColumnSemantics("SkinThickness", "Triceps skin fold thickness", "0-100 mm"),
            ColumnSemantics("Insulin", "2-Hour serum insulin", "0-850 mU/L"),
            ColumnSemantics("BMI", "Body Mass Index", "15-60"),
            ColumnSemantics("DiabetesPedigreeFunction", "Diabetes family history score", "0.08-2.5"),
            ColumnSemantics("Age", "Patient age in years", "20-85"),
            ColumnSemantics("Outcome", "Diabetes diagnosis (1=yes, 0=no)", "0, 1"),
        ],
    )

    proposals = propose_transforms(df, profile, domain, n_sample_rows=5)

    assert isinstance(proposals, list)
    assert len(proposals) > 0
    for p in proposals:
        assert isinstance(p, CleaningProposal)
        assert p.issue_type in {
            "missing_value", "outlier", "duplicate",
            "label_inconsistency", "type_mismatch", "domain_implausible",
        }
        assert p.confidence_tier == UNVERIFIED
        assert len(p.transform_code) > 0
        assert p.affected_count > 0

    issue_types = {p.issue_type for p in proposals}
    print(f"\nProposals generated: {len(proposals)}")
    for p in proposals:
        print(f"  [{p.issue_type}] {p.column}: {p.description[:80]}")
    assert "missing_value" in issue_types or "domain_implausible" in issue_types
