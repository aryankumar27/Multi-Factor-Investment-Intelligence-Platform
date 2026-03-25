"""
MIIP — normaliser.py
Min-max normalization utilities with clamping.
All scores are expressed on a 0–10 scale.
"""

import numpy as np
from config import NORM_BOUNDS


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def minmax_scale(value: float, lo: float, hi: float, invert: bool = False) -> float:
    """
    Scale `value` into [0, 10].
    invert=True means lower raw value → higher score (e.g. debt, volatility).
    """
    if hi == lo:
        return 5.0  # neutral
    value = clamp(value, lo, hi)
    scaled = (value - lo) / (hi - lo)          # → [0,1]
    if invert:
        scaled = 1.0 - scaled
    return round(scaled * 10.0, 4)


def score_metric(name: str, value: float, invert: bool = False) -> float:
    """Score a named metric using pre-defined bounds from config."""
    if value is None or np.isnan(value):
        return 5.0  # neutral fallback
    lo, hi = NORM_BOUNDS.get(name, (0, 1))
    return minmax_scale(value, lo, hi, invert=invert)


def average_scores(scores: dict, weights: dict = None) -> float:
    """Weighted average of a score dict (values 0–10)."""
    if not scores:
        return 5.0
    vals = list(scores.values())
    if weights:
        w = [weights.get(k, 1.0) for k in scores]
        return round(np.average(vals, weights=w), 4)
    return round(np.mean(vals), 4)
