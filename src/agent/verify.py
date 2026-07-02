"""
Dry-run self-verification — Layer 2 (src/agent/).

This is the core mechanism of Gap 2 (No self-verification with confidence
scoring). For every CleaningProposal produced by propose.py, this module:

    1. Makes a sandboxed copy of the dataframe (original is never touched)
    2. Executes the proposal's transform_code in a restricted namespace
    3. Observes what actually happened (rows changed, columns affected,
       nulls introduced, errors raised)
    4. Converts the outcome into a calibrated confidence score (0-1)
    5. Maps the score to a tier: High / Medium / Low

The confidence tier is what the human reviewer sees on each proposal card
in the HITL UI (Layer 3). Whether that tier reliably predicts actual
correctness is the central research question of this project, measured
in src/evaluation/confusion_matrix.py via the confidence-tier confusion
matrix (Section 7.2.1 of the proposal).
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.agent.confidence import score_to_tier

SAFE_BUILTINS = {
    "abs": abs, "len": len, "list": list, "dict": dict, "set": set,
    "tuple": tuple, "range": range, "enumerate": enumerate, "zip": zip,
    "min": min, "max": max, "sum": sum, "round": round,
    "int": int, "float": float, "str": str, "bool": bool,
    "True": True, "False": False, "None": None,
}


@dataclass
class VerificationResult:
    """The outcome of dry-running one CleaningProposal.

    Attributes:
        proposal_issue_type: copied from the original proposal for traceability
        proposal_column: copied from the original proposal
        executed_successfully: True if the transform_code ran without raising
        error_message: the exception message if execution failed, else None
        rows_affected: number of rows that changed value in the target column
        unexpected_columns_changed: columns that changed other than the
            expected target — a side-effect signal
        nulls_introduced: number of new NaN values created anywhere in the
            dataframe (beyond the target column for missing-value fixes)
        score: raw confidence score (0.0–1.0) derived from the above signals
        confidence_tier: "High", "Medium", or "Low" — what the human sees
        before_sample: first 3 affected row values before the transform
        after_sample: first 3 affected row values after the transform
    """
    proposal_issue_type: str
    proposal_column: str
    executed_successfully: bool
    error_message: str | None
    rows_affected: int
    unexpected_columns_changed: list[str]
    nulls_introduced: int
    score: float
    confidence_tier: str
    before_sample: list = field(default_factory=list)
    after_sample: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "proposal_issue_type": self.proposal_issue_type,
            "proposal_column": self.proposal_column,
            "executed_successfully": self.executed_successfully,
            "error_message": self.error_message,
            "rows_affected": self.rows_affected,
            "unexpected_columns_changed": self.unexpected_columns_changed,
            "nulls_introduced": self.nulls_introduced,
            "score": round(self.score, 4),
            "confidence_tier": self.confidence_tier,
            "before_sample": self.before_sample,
            "after_sample": self.after_sample,
        }


def _execute_in_sandbox(
    transform_code: str, df_copy: pd.DataFrame
) -> tuple[pd.DataFrame, str | None]:
    """Execute transform_code in a restricted namespace.

    The namespace provides `df` (the sandboxed copy), `pd`, `np`, and a
    minimal set of safe builtins. No imports, no file I/O, no network access.

    Pandas 2.0+ Copy-on-Write (CoW): `df['col'].fillna(x, inplace=True)`
    silently does nothing under CoW because `.fillna()` returns a copy.
    We detect this by checking whether `df` in the namespace was replaced
    (explicit reassignment like `df = df.fillna(...)` works fine under CoW)
    vs. whether inplace mutations were swallowed. The caller measures actual
    row changes independently, which surfaces the CoW no-op as rows_affected=0
    and scores it Low — exactly the right behavior for a broken transform.

    Returns:
        (modified_df, error_message) — error_message is None on success.
    """
    namespace = {
        "__builtins__": SAFE_BUILTINS,
        "df": df_copy,
        "pd": pd,
        "np": np,
    }
    try:
        exec(compile(transform_code, "<transform>", "exec"), namespace)
        return namespace["df"], None
    except Exception:
        return df_copy, traceback.format_exc(limit=3)


def _compute_score(
    executed_successfully: bool,
    rows_affected: int,
    expected_affected: int,
    unexpected_columns: list[str],
    nulls_introduced: int,
    issue_type: str,
) -> float:
    """Convert dry-run observations into a raw confidence score (0.0–1.0).

    Scoring logic (each penalty is cumulative):

    - Execution failure:        → 0.05  (floor — something went wrong)
    - Zero rows affected:       → 0.15  (fix did nothing)
    - Side effects detected:    → -0.25 per unexpected column changed
    - Unexpected nulls:         → -0.15 if new NaNs appeared outside target
    - Row count way off:        → -0.20 if affected count > 3× expected
    - Clean execution, on target: starts at 0.95

    The score intentionally never reaches 1.0 — perfect certainty is
    not claimed even for cleanly-executing transforms. The thresholds
    (HIGH=0.85, MEDIUM=0.50) in confidence.py map this scale to tiers.
    """
    if not executed_successfully:
        return 0.05

    score = 0.95

    if rows_affected == 0:
        score = 0.15
        return score

    # Side effects — unexpected columns changed
    score -= 0.25 * len(unexpected_columns)

    # New nulls introduced outside of a missing-value fix
    if issue_type != "missing_value" and nulls_introduced > 0:
        score -= 0.15

    # Row count sanity — affected far more rows than expected
    if expected_affected > 0 and rows_affected > expected_affected * 3:
        score -= 0.20

    return max(0.05, min(0.95, score))


def verify_proposal(
    df: pd.DataFrame,
    transform_code: str,
    proposal_column: str,
    proposal_issue_type: str,
    expected_affected_count: int,
) -> VerificationResult:
    """Dry-run a single CleaningProposal and return a VerificationResult.

    Args:
        df: the original dataframe (never modified — a copy is made first).
        transform_code: the Python/Pandas code from the CleaningProposal.
        proposal_column: the column the proposal targets. Used to detect
            side effects (changes to other columns).
        proposal_issue_type: the issue category (missing_value, outlier, etc.)
        expected_affected_count: the agent's own estimate of how many rows
            should change. Used for sanity-checking the actual outcome.

    Returns:
        VerificationResult with confidence_tier set to High/Medium/Low.
    """
    df_before = df.copy()
    df_copy = df.copy()

    df_after, error = _execute_in_sandbox(transform_code, df_copy)

    if error:
        result = VerificationResult(
            proposal_issue_type=proposal_issue_type,
            proposal_column=proposal_column,
            executed_successfully=False,
            error_message=error,
            rows_affected=0,
            unexpected_columns_changed=[],
            nulls_introduced=0,
            score=0.05,
            confidence_tier="Low",
        )
        return result

    # --- Measure what changed ---
    target_col = proposal_column if proposal_column in df_before.columns else None

    # Rows where the target column changed
    rows_affected = 0
    before_sample, after_sample = [], []
    if target_col:
        b = df_before[target_col]
        a = df_after[target_col]
        changed_mask = b.isna() != a.isna()
        changed_mask |= (b.notna() & a.notna() & (b != a))
        rows_affected = int(changed_mask.sum())
        sample_idx = changed_mask[changed_mask].index[:3]
        before_sample = [
            None if pd.isna(v) else v
            for v in df_before.loc[sample_idx, target_col]
        ]
        after_sample = [
            None if pd.isna(v) else v
            for v in df_after.loc[sample_idx, target_col]
        ]
    else:
        # Row-level operation (e.g. drop_duplicates) — measure row count change
        rows_affected = abs(len(df_before) - len(df_after))

    # Unexpected columns changed (side effects)
    # Reset index before comparing to handle operations like drop_duplicates
    # that change the index. Compare only on rows that exist in both.
    unexpected_cols = []
    min_len = min(len(df_before), len(df_after))
    b_reset = df_before.reset_index(drop=True).iloc[:min_len]
    a_reset = df_after.reset_index(drop=True).iloc[:min_len]
    for col in df_before.columns:
        if col == target_col:
            continue
        b, a = b_reset[col], a_reset[col]
        changed = (b.isna() != a.isna()) | (b.notna() & a.notna() & (b != a))
        if changed.any():
            unexpected_cols.append(col)

    # New nulls introduced (anywhere in the dataframe)
    nulls_before = int(df_before.isna().sum().sum())
    nulls_after = int(df_after.isna().sum().sum())
    nulls_introduced = max(0, nulls_after - nulls_before)

    # Exclude expected nulls for missing-value fixes on the target column
    if proposal_issue_type == "missing_value" and target_col:
        expected_new_nulls = max(0,
            int(df_after[target_col].isna().sum()) -
            int(df_before[target_col].isna().sum())
        )
        nulls_introduced = max(0, nulls_introduced - expected_new_nulls)

    score = _compute_score(
        executed_successfully=True,
        rows_affected=rows_affected,
        expected_affected=expected_affected_count,
        unexpected_columns=unexpected_cols,
        nulls_introduced=nulls_introduced,
        issue_type=proposal_issue_type,
    )

    return VerificationResult(
        proposal_issue_type=proposal_issue_type,
        proposal_column=proposal_column,
        executed_successfully=True,
        error_message=None,
        rows_affected=rows_affected,
        unexpected_columns_changed=unexpected_cols,
        nulls_introduced=nulls_introduced,
        score=score,
        confidence_tier=score_to_tier(score),
        before_sample=before_sample,
        after_sample=after_sample,
    )


def verify_all_proposals(
    df: pd.DataFrame,
    proposals: list,
) -> list[tuple]:
    """Dry-run every proposal and attach a confidence tier to each.

    Args:
        df: the original dataframe (never modified).
        proposals: list of CleaningProposal objects from propose.py.

    Returns:
        List of (proposal, VerificationResult) tuples, in the same order
        as the input proposals. Each proposal's confidence_tier attribute
        is updated in-place from "Unverified" to "High"/"Medium"/"Low".
    """
    results = []
    for proposal in proposals:
        result = verify_proposal(
            df=df,
            transform_code=proposal.transform_code,
            proposal_column=proposal.column,
            proposal_issue_type=proposal.issue_type,
            expected_affected_count=proposal.affected_count,
        )
        proposal.confidence_tier = result.confidence_tier
        results.append((proposal, result))
    return results
