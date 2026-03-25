"""
MIIP — fundamental_model.py
Computes the Fundamental Factor Score (0–10).

Methodology:
  Each ratio is scored individually on a 0–10 scale.
  Higher score = more attractive on that dimension.
  Composite = weighted mean of available sub-scores.

Benchmarks used are typical Indian large-cap ranges.
"""

import logging
import numpy as np
from normaliser import minmax_scale

logger = logging.getLogger(__name__)


# ── Scoring functions per ratio ───────────────────────────────────────────────

def score_pe(pe: float) -> float:
    """
    PE ratio.  Lower PE → more undervalued → higher score.
    Range: 0 (very cheap) to 80 (very expensive).
    Neutral: ~25 (Nifty long-run avg).
    """
    if pe is None or pe <= 0:
        return 5.0
    return minmax_scale(pe, 5.0, 60.0, invert=True)


def score_pb(pb: float) -> float:
    """
    PB ratio.  Lower PB → more undervalued → higher score.
    Range: 0.5 → 15.
    """
    if pb is None or pb <= 0:
        return 5.0
    return minmax_scale(pb, 0.5, 15.0, invert=True)


def score_roe(roe: float) -> float:
    """
    Return on Equity.  Higher ROE → better business quality → higher score.
    Range: -20% → 50%.
    """
    if roe is None:
        return 5.0
    return minmax_scale(roe, -0.20, 0.50, invert=False)


def score_debt_equity(de: float) -> float:
    """
    Debt-to-Equity.  Lower D/E → lower leverage risk → higher score.
    Range: 0 → 4 (financial companies excluded from this simple model).
    """
    if de is None or de < 0:
        return 5.0
    return minmax_scale(de, 0.0, 4.0, invert=True)


def score_profit_margin(pm: float) -> float:
    """
    Net Profit Margin.  Higher margin → better efficiency → higher score.
    Range: -10% → 40%.
    """
    if pm is None:
        return 5.0
    return minmax_scale(pm, -0.10, 0.40, invert=False)


def score_revenue_growth(rg: float) -> float:
    """
    Revenue growth (YoY).  Higher → better momentum → higher score.
    Range: -20% → 40%.
    """
    if rg is None:
        return 5.0
    return minmax_scale(rg, -0.20, 0.40, invert=False)


def score_dividend_yield(dy: float) -> float:
    """
    Dividend Yield.  Higher yield = income return, but not always growth.
    Moderate positive.  Range: 0 → 8%.
    """
    if dy is None or dy < 0:
        return 5.0
    return minmax_scale(dy, 0.0, 0.08, invert=False)


# ── Composite ─────────────────────────────────────────────────────────────────

_WEIGHTS = {
    "pe_score":          0.20,
    "pb_score":          0.15,
    "roe_score":         0.25,
    "de_score":          0.15,
    "margin_score":      0.10,
    "rev_growth_score":  0.10,
    "div_yield_score":   0.05,
}


def compute_fundamental_score(data: dict) -> dict:
    """
    Parameters
    ----------
    data : dict
        Keys: pe_ratio, pb_ratio, roe, debt_equity,
              profit_margin, revenue_growth, dividend_yield

    Returns
    -------
    dict with keys: score (0-10), sub_scores, commentary
    """
    sub = {}
    commentary = []

    def _fmt(label, val, pct=False):
        if val is None:
            return f"{label}: N/A"
        return f"{label}: {val*100:.1f}%" if pct else f"{label}: {val:.2f}"

    pe   = data.get("pe_ratio")
    pb   = data.get("pb_ratio")
    roe  = data.get("roe")
    de   = data.get("debt_equity")
    pm   = data.get("profit_margin")
    rg   = data.get("revenue_growth")
    dy   = data.get("dividend_yield")

    sub["pe_score"]         = score_pe(pe)
    sub["pb_score"]         = score_pb(pb)
    sub["roe_score"]        = score_roe(roe)
    sub["de_score"]         = score_debt_equity(de)
    sub["margin_score"]     = score_profit_margin(pm)
    sub["rev_growth_score"] = score_revenue_growth(rg)
    sub["div_yield_score"]  = score_dividend_yield(dy)

    commentary = [
        _fmt("PE",        pe),
        _fmt("PB",        pb),
        _fmt("ROE",       roe,  pct=True),
        _fmt("D/E",       de),
        _fmt("Margin",    pm,   pct=True),
        _fmt("Rev Growth",rg,   pct=True),
        _fmt("Div Yield", dy,   pct=True),
    ]

    # Weighted composite (skip neutral 5.0 for missing fields to avoid drag)
    total_w  = sum(_WEIGHTS[k] for k in sub)
    score    = sum(sub[k] * _WEIGHTS[k] for k in sub) / total_w
    score    = round(score, 4)

    # Earnings quality bonus / penalty
    if roe is not None and de is not None:
        if roe > 0.20 and de < 0.5:
            score = min(10.0, score + 0.3)    # high-quality compounder bonus
        elif roe < 0.05 and de > 2.0:
            score = max(0.0,  score - 0.5)    # leveraged / low-profitability penalty

    logger.info("Fundamental score: %.2f | sub=%s", score, sub)
    return {
        "score":      score,
        "sub_scores": sub,
        "commentary": " | ".join(commentary),
        "raw_data":   data,
    }
