"""Tests for the dry-run self-verification module (src/agent/verify.py).

All tests are pure unit tests - no API calls, no external dependencies.

Run with: pytest tests/test_verify.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.verify import (
    VerificationResult,
    _compute_score,
    _execute_in_sandbox,
    verify_all_proposals,
    verify_proposal,
)

# CoW-safe transform codes (pandas 2.0+)
CODE_FILL_GLUCOSE = "df['Glucose'] = df['Glucose'].fillna(df['Glucose'].median())"
CODE_FILL_BMI     = "df['BMI'] = df['BMI'].fillna(df['BMI'].median())"
CODE_FILL_999     = "df['Glucose'] = df['Glucose'].fillna(999)"
CODE_FILL_BROKEN  = "df['Glucose'].fillna(df['Glucose'].median(), inplace=True)"
CODE_FILL_NONEXISTENT = "df['NonExistentColumn'].fillna(0, inplace=True)"
CODE_CAP_AGE      = "df.loc[df['Age'] > 200, 'Age'] = df['Age'].median()"
CODE_DROP_DUPES   = "df.drop_duplicates(inplace=True)"
CODE_SIDE_EFFECT  = CODE_FILL_GLUCOSE + "\ndf['BMI'] = 0"


def make_df():
    return pd.DataFrame({
        "Glucose":       [148.0, 85.0, None, 89.0, 137.0],
        "BloodPressure": [72.0,  66.0,  64.0,  66.0,   0.0],
        "BMI":           [33.6, None,  23.3,  28.1,  43.1],
        "Age":           [50.0,  31.0,  32.0,  21.0,  33.0],
    })


class TestSandboxExecution:

    def test_clean_code_runs_without_error(self):
        df = make_df()
        result, err = _execute_in_sandbox(CODE_FILL_GLUCOSE, df)
        assert err is None
        assert result["Glucose"].isna().sum() == 0

    def test_bad_code_returns_error_message(self):
        df = make_df()
        _, err = _execute_in_sandbox("raise ValueError('boom')", df)
        assert err is not None
        assert "ValueError" in err

    def test_sandbox_blocks_import(self):
        df = make_df()
        _, err = _execute_in_sandbox("import os; os.system('ls')", df)
        assert err is not None

    def test_sandbox_blocks_open(self):
        df = make_df()
        _, err = _execute_in_sandbox("open('/etc/passwd')", df)
        assert err is not None

    def test_original_df_not_modified(self):
        df = make_df()
        original_nulls = df["Glucose"].isna().sum()
        _execute_in_sandbox(CODE_FILL_999, df.copy())
        assert df["Glucose"].isna().sum() == original_nulls

    def test_cow_broken_inplace_detected_as_zero_rows_affected(self):
        """Pandas CoW: df['col'].fillna(x, inplace=True) silently does nothing.
        The sandbox does not error — but rows_affected=0 surfaces it as Low."""
        df = make_df()
        result = verify_proposal(
            df=df,
            transform_code=CODE_FILL_BROKEN,
            proposal_column="Glucose",
            proposal_issue_type="missing_value",
            expected_affected_count=1,
        )
        assert result.rows_affected == 0
        assert result.confidence_tier == "Low"


class TestScoreComputation:

    def test_failed_execution_gives_floor_score(self):
        score = _compute_score(False, 0, 0, [], 0, "missing_value")
        assert score == 0.05

    def test_zero_rows_affected_gives_low_score(self):
        score = _compute_score(True, 0, 5, [], 0, "missing_value")
        assert score < 0.5

    def test_clean_execution_gives_high_score(self):
        score = _compute_score(True, 5, 5, [], 0, "missing_value")
        assert score >= 0.85

    def test_side_effects_reduce_score(self):
        clean = _compute_score(True, 5, 5, [], 0, "outlier")
        with_side_effects = _compute_score(True, 5, 5, ["BMI", "Age"], 0, "outlier")
        assert with_side_effects < clean

    def test_unexpected_nulls_reduce_score_for_non_missing_fix(self):
        clean = _compute_score(True, 5, 5, [], 0, "outlier")
        with_nulls = _compute_score(True, 5, 5, [], 3, "outlier")
        assert with_nulls < clean

    def test_unexpected_nulls_do_not_penalize_missing_value_fix(self):
        score_with = _compute_score(True, 5, 5, [], 5, "missing_value")
        score_without = _compute_score(True, 5, 5, [], 0, "missing_value")
        assert score_with == score_without

    def test_score_clamped_between_floor_and_ceiling(self):
        s1 = _compute_score(True, 5, 5, ["a", "b", "c", "d"], 5, "outlier")
        assert s1 >= 0.05
        s2 = _compute_score(True, 5, 5, [], 0, "missing_value")
        assert s2 <= 0.95


class TestVerifyProposal:

    def test_successful_missing_value_fix_gets_high_tier(self):
        df = make_df()
        result = verify_proposal(
            df=df,
            transform_code=CODE_FILL_GLUCOSE,
            proposal_column="Glucose",
            proposal_issue_type="missing_value",
            expected_affected_count=1,
        )
        assert result.executed_successfully is True
        assert result.rows_affected == 1
        assert result.confidence_tier == "High"
        assert result.error_message is None

    def test_successful_outlier_fix_gets_high_tier(self):
        df = make_df()
        result = verify_proposal(
            df=df,
            transform_code=CODE_CAP_AGE,
            proposal_column="Age",
            proposal_issue_type="outlier",
            expected_affected_count=0,
        )
        assert result.executed_successfully is True
        assert result.confidence_tier in {"High", "Medium", "Low"}

    def test_broken_code_gets_low_tier(self):
        df = make_df()
        result = verify_proposal(
            df=df,
            transform_code=CODE_FILL_NONEXISTENT,
            proposal_column="Glucose",
            proposal_issue_type="missing_value",
            expected_affected_count=1,
        )
        assert result.executed_successfully is False
        assert result.confidence_tier == "Low"
        assert result.error_message is not None

    def test_side_effect_detected(self):
        df = make_df()
        result = verify_proposal(
            df=df,
            transform_code=CODE_SIDE_EFFECT,
            proposal_column="Glucose",
            proposal_issue_type="missing_value",
            expected_affected_count=1,
        )
        assert "BMI" in result.unexpected_columns_changed
        assert result.score < 0.95

    def test_before_and_after_samples_captured(self):
        df = make_df()
        result = verify_proposal(
            df=df,
            transform_code=CODE_FILL_GLUCOSE,
            proposal_column="Glucose",
            proposal_issue_type="missing_value",
            expected_affected_count=1,
        )
        assert len(result.before_sample) > 0
        assert len(result.after_sample) > 0
        assert result.before_sample[0] is None
        assert result.after_sample[0] is not None

    def test_original_dataframe_unchanged_after_verify(self):
        df = make_df()
        original_null_count = df.isna().sum().sum()
        verify_proposal(
            df=df,
            transform_code=CODE_FILL_999,
            proposal_column="Glucose",
            proposal_issue_type="missing_value",
            expected_affected_count=1,
        )
        assert df.isna().sum().sum() == original_null_count

    def test_duplicate_removal_measured_as_row_count_change(self):
        df = pd.DataFrame({"A": [1, 2, 1, 3], "B": [4, 5, 4, 6]})
        result = verify_proposal(
            df=df,
            transform_code=CODE_DROP_DUPES,
            proposal_column="<row>",
            proposal_issue_type="duplicate",
            expected_affected_count=1,
        )
        assert result.executed_successfully is True
        assert result.rows_affected == 1


class TestVerifyAllProposals:

    def test_updates_confidence_tier_on_proposals(self):
        df = make_df()

        p1 = MagicMock()
        p1.transform_code = CODE_FILL_GLUCOSE
        p1.column = "Glucose"
        p1.issue_type = "missing_value"
        p1.affected_count = 1
        p1.confidence_tier = "Unverified"

        p2 = MagicMock()
        p2.transform_code = "this is invalid python !!!"
        p2.column = "BMI"
        p2.issue_type = "missing_value"
        p2.affected_count = 1
        p2.confidence_tier = "Unverified"

        results = verify_all_proposals(df, [p1, p2])

        assert len(results) == 2
        assert p1.confidence_tier == "High"
        assert p2.confidence_tier == "Low"

    def test_returns_proposal_result_pairs(self):
        df = make_df()
        p = MagicMock()
        p.transform_code = CODE_FILL_BMI
        p.column = "BMI"
        p.issue_type = "missing_value"
        p.affected_count = 1

        pairs = verify_all_proposals(df, [p])
        assert len(pairs) == 1
        proposal, result = pairs[0]
        assert proposal is p
        assert isinstance(result, VerificationResult)
