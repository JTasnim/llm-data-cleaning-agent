"""
Layer 1 — Data Profiler

Scans a dataframe and produces a structured profile: column types, null
rates, unique-value counts, statistical summaries, and detected anomalies
(mixed types, outlier z-scores, duplicate rows).

This is the input the Layer 2 LLM Reasoning Agent (src/agent/) consumes to
infer domain/semantics and propose cleaning transformations. The agent
should never need to scan the raw dataframe itself — everything it needs to
reason about should be summarized here.

Great Expectations integration is planned as a follow-up layer on top of
this pure-Pandas profiler (see docs/architecture.md); this module is
self-contained and dependency-light so it can be developed and tested first.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_count: int
    null_rate: float
    unique_count: int
    is_numeric: bool
    is_mixed_type: bool = False
    # Numeric-only fields (None for non-numeric columns)
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    outlier_count: int = 0
    outlier_row_indices: list[int] = field(default_factory=list)


@dataclass
class DatasetProfile:
    n_rows: int
    n_columns: int
    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    duplicate_row_count: int = 0
    duplicate_row_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a plain dict — the form the LLM agent will consume
        (e.g. serialized into a prompt as JSON).
        """
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "duplicate_row_count": self.duplicate_row_count,
            "duplicate_row_indices": self.duplicate_row_indices[:20],  # cap for prompt size
            "columns": {
                name: {
                    "dtype": col.dtype,
                    "null_count": col.null_count,
                    "null_rate": round(col.null_rate, 4),
                    "unique_count": col.unique_count,
                    "is_numeric": col.is_numeric,
                    "is_mixed_type": col.is_mixed_type,
                    "mean": col.mean,
                    "std": col.std,
                    "min": col.min,
                    "max": col.max,
                    "outlier_count": col.outlier_count,
                    "outlier_row_indices": col.outlier_row_indices[:20],
                }
                for name, col in self.columns.items()
            },
        }


def _is_mixed_type(series: pd.Series) -> bool:
    """Detect a numeric-looking column that actually contains non-numeric
    strings (e.g. "N/A", "—") mixed in with numbers. Only meaningful for
    non-numeric-dtype columns; a column pandas already parsed as int/float
    can't be mixed-type by definition.
    """
    if pd.api.types.is_numeric_dtype(series):
        return False

    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    numeric_like = pd.to_numeric(non_null, errors="coerce")
    n_numeric = numeric_like.notna().sum()
    # "Mixed" means some but not all values look numeric — a fully numeric
    # string column (e.g. all strings of digits) is a type issue but not a
    # *mixed* one, and a fully non-numeric column is just categorical.
    return bool(0 < n_numeric < len(non_null))


def _detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> list[int]:
    """Return row indices flagged as outliers via the IQR method.

    More robust than z-score for small samples or samples containing a
    single extreme value, since a z-score outlier check can be defeated by
    the very outlier inflating the standard deviation it's measured against.
    Returns [] for columns with fewer than 4 non-null values (IQR is not
    meaningful below that) or zero IQR (no spread to compare against).
    """
    non_null = series.dropna()
    if len(non_null) < 4:
        return []

    q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return []

    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (non_null < lower) | (non_null > upper)
    return non_null.index[mask].tolist()


def _detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> list[int]:
    """Return row indices where the absolute z-score exceeds the threshold.

    Uses population std (ddof=0); returns [] for columns with fewer than 2
    non-null values or zero variance, where z-scores are undefined/meaningless.
    """
    non_null = series.dropna()
    if len(non_null) < 2:
        return []

    std = non_null.std()
    if std == 0 or pd.isna(std):
        return []

    mean = non_null.mean()
    z_scores = (non_null - mean).abs() / std
    return z_scores[z_scores > threshold].index.tolist()


def _detect_outliers(series: pd.Series, zscore_threshold: float = 3.0) -> list[int]:
    """Combine z-score and IQR detection: a value is flagged if either
    method flags it. IQR catches single dominant outliers that inflate
    their own z-score denominator; z-score catches outliers in larger,
    well-behaved distributions that IQR's fixed 1.5x multiplier might miss.
    """
    z_outliers = set(_detect_outliers_zscore(series, threshold=zscore_threshold))
    iqr_outliers = set(_detect_outliers_iqr(series))
    return sorted(z_outliers | iqr_outliers)


def _profile_column(series: pd.Series, name: str) -> ColumnProfile:
    null_count = int(series.isna().sum())
    n = len(series)
    is_numeric = bool(pd.api.types.is_numeric_dtype(series))
    mixed = _is_mixed_type(series)

    profile = ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        null_count=null_count,
        null_rate=null_count / n if n else 0.0,
        unique_count=int(series.nunique(dropna=True)),
        is_numeric=is_numeric,
        is_mixed_type=mixed,
    )

    if is_numeric:
        non_null = series.dropna()
        if len(non_null) > 0:
            profile.mean = float(non_null.mean())
            profile.std = float(non_null.std()) if len(non_null) > 1 else 0.0
            profile.min = float(non_null.min())
            profile.max = float(non_null.max())
        outlier_idx = _detect_outliers(series)
        profile.outlier_count = len(outlier_idx)
        profile.outlier_row_indices = outlier_idx

    return profile


def profile_dataset(df: pd.DataFrame, outlier_zscore_threshold: float = 3.0) -> DatasetProfile:
    """Produce a structured profile of a dataframe.

    Args:
        df: the dataframe to profile (read-only — never modified).
        outlier_zscore_threshold: |z| above which a numeric value is flagged
            as an outlier. 3.0 is the conventional default.

    Returns:
        A DatasetProfile with per-column stats and dataset-level anomalies
        (duplicate rows). Call .to_dict() to get a JSON-serializable form
        suitable for embedding in an LLM prompt.
    """
    duplicate_mask = df.duplicated(keep="first")
    duplicate_indices = df.index[duplicate_mask].tolist()

    columns = {
        col: _profile_column(df[col], col) for col in df.columns
    }

    return DatasetProfile(
        n_rows=len(df),
        n_columns=len(df.columns),
        columns=columns,
        duplicate_row_count=len(duplicate_indices),
        duplicate_row_indices=duplicate_indices,
    )
