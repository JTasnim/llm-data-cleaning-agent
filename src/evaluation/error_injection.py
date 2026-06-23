"""
Synthetic error injection — Section 7.1 / 6.2 of the project proposal.

Takes a clean, real-world dataset and deliberately injects a controlled set
of known errors (missing values, inconsistent category labels, numeric
outliers, and duplicate rows), recording a ground-truth ledger of every
change made.

This is necessary because ground-truth evaluation of recall/precision
requires knowing the "correct" answer in advance — something that real,
naturally-occurring messy data does not provide. The original (pre-injection)
dataset serves as the answer key against which an agent's detected issues
and proposed fixes are later scored.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class InjectedError:
    """A single record in the ground-truth error ledger."""

    row_index: int
    column: str
    error_type: str  # "missing_value" | "outlier" | "label_inconsistency" | "duplicate"
    original_value: object
    corrupted_value: object


@dataclass
class CorruptionResult:
    corrupted_df: pd.DataFrame
    ledger: list[InjectedError] = field(default_factory=list)

    def ledger_as_dataframe(self) -> pd.DataFrame:
        """Convert the ledger to a DataFrame for easy CSV export."""
        if not self.ledger:
            return pd.DataFrame(
                columns=["row_index", "column", "error_type", "original_value", "corrupted_value"]
            )
        return pd.DataFrame([e.__dict__ for e in self.ledger])


def inject_missing_values(
    df: pd.DataFrame, columns: list[str], rate: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[InjectedError]]:
    """Replace a fraction of values in the given columns with NaN.

    Args:
        df: dataframe to corrupt (not modified in place).
        columns: column names to target.
        rate: fraction of rows in each column to null out (0-1).
        rng: a seeded numpy random Generator, shared across injection steps
            so the whole corruption run is reproducible from one seed.

    Returns:
        (corrupted_df, ledger_entries)
    """
    df = df.copy()
    ledger: list[InjectedError] = []

    for col in columns:
        if col not in df.columns:
            continue
        n_to_null = int(len(df) * rate)
        # Only target rows that aren't already null, so we don't double-count.
        candidate_idx = df.index[df[col].notna()]
        if len(candidate_idx) == 0:
            continue
        n_to_null = min(n_to_null, len(candidate_idx))
        chosen_idx = rng.choice(candidate_idx, size=n_to_null, replace=False)

        for idx in chosen_idx:
            original = df.at[idx, col]
            df.at[idx, col] = np.nan
            ledger.append(
                InjectedError(
                    row_index=int(idx),
                    column=col,
                    error_type="missing_value",
                    original_value=original,
                    corrupted_value=np.nan,
                )
            )

    return df, ledger


def inject_label_inconsistencies(
    df: pd.DataFrame,
    column: str,
    variant_map: dict[str, list[str]],
    rng: np.random.Generator,
    fraction: float = 0.5,
) -> tuple[pd.DataFrame, list[InjectedError]]:
    """Replace some occurrences of canonical category labels with
    inconsistent variants (e.g. "USA" -> "U.S.A." / "United States").

    Args:
        df: dataframe to corrupt.
        column: categorical column to target.
        variant_map: maps a canonical label to a list of inconsistent
            spellings/variants to substitute in, e.g.
            {"USA": ["U.S.A.", "United States", "us"]}.
        rng: shared seeded random Generator.
        fraction: fraction of matching rows (per canonical label) to
            replace with a randomly chosen variant.

    Returns:
        (corrupted_df, ledger_entries)
    """
    if column not in df.columns:
        return df.copy(), []

    df = df.copy()
    ledger: list[InjectedError] = []

    for canonical, variants in variant_map.items():
        if not variants:
            continue
        matching_idx = df.index[df[column] == canonical]
        if len(matching_idx) == 0:
            continue
        n_to_change = int(len(matching_idx) * fraction)
        if n_to_change == 0:
            continue
        chosen_idx = rng.choice(matching_idx, size=n_to_change, replace=False)

        for idx in chosen_idx:
            variant = rng.choice(variants)
            df.at[idx, column] = variant
            ledger.append(
                InjectedError(
                    row_index=int(idx),
                    column=column,
                    error_type="label_inconsistency",
                    original_value=canonical,
                    corrupted_value=variant,
                )
            )

    return df, ledger


def inject_outliers(
    df: pd.DataFrame,
    column: str,
    n: int,
    rng: np.random.Generator,
    multiplier: float = 10.0,
) -> tuple[pd.DataFrame, list[InjectedError]]:
    """Inject implausible numeric outliers into a column by multiplying a
    sample of existing values by a large factor (e.g. age = 999).

    Args:
        df: dataframe to corrupt.
        column: numeric column to target.
        n: number of outliers to inject.
        rng: shared seeded random Generator.
        multiplier: factor applied to the original value to produce an
            implausible outlier.

    Returns:
        (corrupted_df, ledger_entries)
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return df.copy(), []

    df = df.copy()
    ledger: list[InjectedError] = []

    candidate_idx = df.index[df[column].notna()]
    n = min(n, len(candidate_idx))
    if n == 0:
        return df, ledger

    chosen_idx = rng.choice(candidate_idx, size=n, replace=False)
    for idx in chosen_idx:
        original = df.at[idx, column]
        corrupted = original * multiplier if original != 0 else multiplier
        df.at[idx, column] = corrupted
        ledger.append(
            InjectedError(
                row_index=int(idx),
                column=column,
                error_type="outlier",
                original_value=original,
                corrupted_value=corrupted,
            )
        )

    return df, ledger


def inject_duplicates(
    df: pd.DataFrame, n: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[InjectedError]]:
    """Append n duplicate rows (exact copies of randomly chosen existing
    rows) to the end of the dataframe.

    Returns:
        (corrupted_df, ledger_entries) — the ledger records the *new* row
        index of each duplicate and which original row_index it copies, via
        the corrupted_value field.
    """
    df = df.copy()
    ledger: list[InjectedError] = []

    n = min(n, len(df))
    if n == 0:
        return df, ledger

    source_idx = rng.choice(df.index, size=n, replace=False)
    duplicated_rows = df.loc[source_idx].copy()
    df = pd.concat([df, duplicated_rows], ignore_index=True)

    new_row_start = len(df) - n
    for offset, orig_idx in enumerate(source_idx):
        new_idx = new_row_start + offset
        ledger.append(
            InjectedError(
                row_index=int(new_idx),
                column="<row>",
                error_type="duplicate",
                original_value=int(orig_idx),
                corrupted_value=int(new_idx),
            )
        )

    return df, ledger


def corrupt_dataset(
    df: pd.DataFrame,
    missing_columns: list[str] | None = None,
    missing_rate: float = 0.05,
    label_column: str | None = None,
    label_variant_map: dict[str, list[str]] | None = None,
    outlier_column: str | None = None,
    n_outliers: int = 10,
    n_duplicates: int = 10,
    seed: int = 42,
) -> CorruptionResult:
    """Apply the full, controlled corruption pipeline to a clean dataset and
    return the corrupted dataframe plus a combined ground-truth ledger.

    All injection steps share a single seeded random Generator so the entire
    corruption run is reproducible end-to-end from one seed value.
    """
    rng = np.random.default_rng(seed)
    combined_ledger: list[InjectedError] = []
    working_df = df.copy()

    if missing_columns:
        working_df, ledger = inject_missing_values(working_df, missing_columns, missing_rate, rng)
        combined_ledger.extend(ledger)

    if label_column and label_variant_map:
        working_df, ledger = inject_label_inconsistencies(working_df, label_column, label_variant_map, rng)
        combined_ledger.extend(ledger)

    if outlier_column:
        working_df, ledger = inject_outliers(working_df, outlier_column, n_outliers, rng)
        combined_ledger.extend(ledger)

    if n_duplicates:
        working_df, ledger = inject_duplicates(working_df, n_duplicates, rng)
        combined_ledger.extend(ledger)

    return CorruptionResult(corrupted_df=working_df, ledger=combined_ledger)
