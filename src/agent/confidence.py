"""
Confidence scoring — the core mechanism for this project's research focus
(Gap 2: No self-verification with confidence scoring).

Maps a dry-run self-verification outcome to a confidence tier (High /
Medium / Low). The calibration of this mapping — i.e., whether High-tier
proposals are actually more likely to be correct than Low-tier ones — is
what src/evaluation/confusion_matrix.py measures via the confidence-tier
confusion matrix (Section 7.2.1 of the proposal).
"""
from __future__ import annotations

import os

HIGH_THRESHOLD = float(os.getenv("CONFIDENCE_HIGH_THRESHOLD", "0.85"))
MEDIUM_THRESHOLD = float(os.getenv("CONFIDENCE_MEDIUM_THRESHOLD", "0.5"))


def score_to_tier(verification_score: float) -> str:
    """Convert a raw self-verification score (0-1) into a confidence tier.

    Args:
        verification_score: estimated probability that the proposed fix is
            correct and free of side effects, derived from the dry-run
            outcome in src/agent/verify.py.

    Returns:
        One of "High", "Medium", "Low".
    """
    if verification_score >= HIGH_THRESHOLD:
        return "High"
    if verification_score >= MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def dry_run_to_score(dry_run_passed: bool, side_effects_detected: bool) -> float:
    """Placeholder heuristic combining dry-run outcome into a raw score.

    TODO: replace with the calibrated scoring function from verify.py
    once the full self-verification loop is integrated. This naive version
    is kept here so the confidence module can be tested independently
    before verify.py is wired in.
    """
    if not dry_run_passed:
        return 0.1
    if side_effects_detected:
        return 0.4
    return 0.9
