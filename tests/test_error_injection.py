"""Tests for the synthetic error-injection pipeline.

Run with: pytest tests/test_error_injection.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.error_injection import (
    corrupt_dataset,
    inject_duplicates,
    inject_missing_values,
    inject_outliers,
)


def make_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "country": ["USA"] * 10,
            "score": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
        }
    )


def test_inject_missing_values_count_matches_rate():
    df = make_sample_df()
    rng = np.random.default_rng(0)
    corrupted, ledger = inject_missing_values(df, ["age"], rate=0.3, rng=rng)
    assert corrupted["age"].isna().sum() == 3
    assert len(ledger) == 3
    assert all(e.error_type == "missing_value" for e in ledger)


def test_inject_outliers_changes_values_and_records_originals():
    df = make_sample_df()
    rng = np.random.default_rng(0)
    corrupted, ledger = inject_outliers(df, "score", n=2, rng=rng, multiplier=10.0)
    assert len(ledger) == 2
    for entry in ledger:
        assert corrupted.at[entry.row_index, "score"] == entry.original_value * 10.0


def test_inject_duplicates_increases_row_count():
    df = make_sample_df()
    rng = np.random.default_rng(0)
    corrupted, ledger = inject_duplicates(df, n=4, rng=rng)
    assert len(corrupted) == len(df) + 4
    assert len(ledger) == 4


def test_corrupt_dataset_is_reproducible_with_same_seed():
    df = make_sample_df()
    result_a = corrupt_dataset(df, missing_columns=["age"], missing_rate=0.2, n_duplicates=2, seed=7)
    result_b = corrupt_dataset(df, missing_columns=["age"], missing_rate=0.2, n_duplicates=2, seed=7)
    assert result_a.corrupted_df.equals(result_b.corrupted_df)
    assert len(result_a.ledger) == len(result_b.ledger)


def test_corrupt_dataset_ledger_as_dataframe_has_expected_columns():
    df = make_sample_df()
    result = corrupt_dataset(df, missing_columns=["age"], missing_rate=0.2, seed=1)
    ledger_df = result.ledger_as_dataframe()
    assert list(ledger_df.columns) == [
        "row_index",
        "column",
        "error_type",
        "original_value",
        "corrupted_value",
    ]
