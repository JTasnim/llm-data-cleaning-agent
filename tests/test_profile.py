"""Tests for the Layer 1 data profiler.

Run with: pytest tests/test_profile.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.profiler.profile import profile_dataset


def test_profile_reports_correct_shape():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    profile = profile_dataset(df)
    assert profile.n_rows == 3
    assert profile.n_columns == 2


def test_profile_detects_null_rate():
    df = pd.DataFrame({"a": [1, None, 3, None]})
    profile = profile_dataset(df)
    col = profile.columns["a"]
    assert col.null_count == 2
    assert col.null_rate == 0.5


def test_profile_detects_duplicate_rows():
    df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
    profile = profile_dataset(df)
    assert profile.duplicate_row_count == 1
    assert profile.duplicate_row_indices == [2]


def test_profile_detects_mixed_type_column():
    df = pd.DataFrame({"a": ["1", "2", "N/A", "4"]})
    profile = profile_dataset(df)
    assert profile.columns["a"].is_mixed_type is True


def test_profile_does_not_flag_fully_numeric_object_column_as_mixed():
    df = pd.DataFrame({"a": ["1", "2", "3", "4"]})
    profile = profile_dataset(df)
    assert profile.columns["a"].is_mixed_type is False


def test_profile_does_not_flag_fully_categorical_column_as_mixed():
    df = pd.DataFrame({"a": ["red", "green", "blue"]})
    profile = profile_dataset(df)
    assert profile.columns["a"].is_mixed_type is False


def test_profile_detects_numeric_outlier():
    # 19 normal values clustered near 50, one wildly off at 999 — large
    # enough sample that the single outlier doesn't dominate the std.
    values = [48, 49, 50, 51, 52, 49, 50, 51, 48, 50,
              49, 51, 50, 48, 52, 49, 50, 51, 50, 999]
    df = pd.DataFrame({"a": values})
    profile = profile_dataset(df)
    col = profile.columns["a"]
    assert col.outlier_count == 1
    assert col.outlier_row_indices == [19]


def test_profile_iqr_catches_outlier_that_dominates_small_sample_std():
    # With only 10 points, one extreme value inflates its own z-score
    # denominator enough to evade z-score detection — IQR should still
    # catch it, demonstrating why the profiler combines both methods.
    values = [50, 51, 49, 50, 52, 48, 50, 51, 49, 999]
    df = pd.DataFrame({"a": values})
    profile = profile_dataset(df)
    col = profile.columns["a"]
    assert col.outlier_count == 1
    assert col.outlier_row_indices == [9]


def test_profile_handles_zero_variance_column_without_error():
    df = pd.DataFrame({"a": [5, 5, 5, 5]})
    profile = profile_dataset(df)
    assert profile.columns["a"].outlier_count == 0


def test_profile_to_dict_is_json_serializable_shape():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    profile = profile_dataset(df)
    d = profile.to_dict()
    assert "columns" in d
    assert "a" in d["columns"]
    assert d["columns"]["a"]["null_count"] == 1
