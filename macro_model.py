"""
MIIP — macro_model.py
Computes the Macro Factor Score (0–10).

Logic mirrors what a macro desk analyst would assess:
  • Nifty trend      → market regime
  • USDINR trend     → currency pressure
  • Crude trend      → input cost / CAD risk
  • India VIX level  → risk-off vs risk-on environment
"""

import logging
import numpy as np
import pandas as pd
from normaliser import minmax_scale

logger = logging.getLogger(__name__)


def _rolling_return(series: pd.Series, window: int) -> float:
    """Return over the last `window` trading days."""
    if len(series) < window + 1:
        return 0.0
    return float((series.iloc[-1] / series.iloc[-(window + 1)]) - 1.0)


def _trend_score(ret: float, lo: float = -0.15, hi: float = 0.15) -> float:
    """Convert a return into 0–10 score; higher return → higher score."""
    return minmax_scale(ret, lo, hi, invert=False)


def _vix_score(vix: float) -> float:
    """Lower VIX → lower fear → better macro env → higher score."""
    return minmax_scale(vix, 10.0, 40.0, invert=True)


def compute_macro_score(macro_df: pd.DataFrame) -> dict:
    """
    Parameters
    ----------
    macro_df : pd.DataFrame
        Columns: nifty, usdinr, crude, india_vix  (daily or monthly).

    Returns
    -------
    dict with keys: score (0-10), sub_scores, commentary
    """
    if macro_df is None or macro_df.empty:
        logger.warning("No macro data; returning neutral score")
        return {"score": 5.0, "sub_scores": {}, "commentary": "No macro data available."}

    result = {}
    commentary = []

    # ── 1. Nifty 3-month trend ───────────────────────────────────────────────
    if "nifty" in macro_df.columns and macro_df["nifty"].notna().sum() > 63:
        nifty_ret = _rolling_return(macro_df["nifty"].dropna(), 63)
        result["nifty_trend"] = _trend_score(nifty_ret)
        direction = "↑" if nifty_ret > 0 else "↓"
        commentary.append(f"Nifty 3M return: {nifty_ret:+.1%} {direction}")
    else:
        result["nifty_trend"] = 5.0

    # ── 2. USDINR trend (rupee depreciation = bad for equities) ─────────────
    if "usdinr" in macro_df.columns and macro_df["usdinr"].notna().sum() > 63:
        fx_ret = _rolling_return(macro_df["usdinr"].dropna(), 63)
        # Rupee weakening (USDINR up) is negative for equity → invert
        result["fx_stability"] = _trend_score(-fx_ret)
        commentary.append(f"USDINR 3M move: {fx_ret:+.1%} (neg = rupee weakness)")
    else:
        result["fx_stability"] = 5.0

    # ── 3. Crude oil 3-month trend (higher crude → cost pressure → bad) ─────
    if "crude" in macro_df.columns and macro_df["crude"].notna().sum() > 63:
        crude_ret = _rolling_return(macro_df["crude"].dropna(), 63)
        result["crude_impact"] = _trend_score(-crude_ret)   # invert
        commentary.append(f"Crude 3M return: {crude_ret:+.1%} (rising crude = headwind)")
    else:
        result["crude_impact"] = 5.0

    # ── 4. India VIX ─────────────────────────────────────────────────────────
    if "india_vix" in macro_df.columns and macro_df["india_vix"].notna().any():
        latest_vix = float(macro_df["india_vix"].dropna().iloc[-1])
        result["vix_environment"] = _vix_score(latest_vix)
        commentary.append(f"India VIX: {latest_vix:.1f}")
    else:
        result["vix_environment"] = 5.0

    # ── Composite ─────────────────────────────────────────────────────────────
    weights = {
        "nifty_trend":    0.35,
        "fx_stability":   0.25,
        "crude_impact":   0.20,
        "vix_environment":0.20,
    }
    score = float(np.average(list(result.values()), weights=[weights[k] for k in result]))
    score = round(score, 4)

    logger.info("Macro score: %.2f | sub=%s", score, result)
    return {
        "score":       score,
        "sub_scores":  result,
        "commentary":  " | ".join(commentary),
        "weights_used": weights,
    }
