"""
MIIP — sentiment_model.py
Computes the Sentiment / Technical Factor Score (0–10).

Factors:
  • Price vs 50-day MA   (golden/death cross proxy)
  • Price vs 200-day MA  (long-term trend)
  • 1-month momentum
  • 3-month momentum
  • 6-month momentum
  • Relative Strength vs Nifty (3M)
  • Volume trend
"""

import logging
import numpy as np
import pandas as pd
from normaliser import minmax_scale

logger = logging.getLogger(__name__)


def _ret(prices: pd.Series, days: int) -> float:
    if len(prices) < days + 1:
        return 0.0
    return float(prices.iloc[-1] / prices.iloc[-(days + 1)] - 1.0)


def _ma_spread(prices: pd.Series, window: int) -> float:
    """(price - MA) / MA — positive = price above MA."""
    if len(prices) < window:
        return 0.0
    ma = prices.rolling(window).mean().iloc[-1]
    if ma == 0:
        return 0.0
    return float((prices.iloc[-1] - ma) / ma)


def _volume_trend(volume: pd.Series, short: int = 20, long: int = 60) -> float:
    """Ratio of short-term avg volume to long-term avg volume."""
    if len(volume) < long:
        return 1.0
    st = volume.rolling(short).mean().iloc[-1]
    lt = volume.rolling(long).mean().iloc[-1]
    return float(st / lt) if lt > 0 else 1.0


def _relative_strength(stock_prices: pd.Series, bench_prices: pd.Series, days: int = 63) -> float:
    """Return of stock minus return of benchmark over `days`."""
    if bench_prices is None or len(bench_prices) < days + 1:
        return 0.0
    s_ret = _ret(stock_prices, days)
    b_ret = _ret(bench_prices, days)
    return s_ret - b_ret


# ── Scoring helpers ────────────────────────────────────────────────────────────

def score_ma_spread(spread: float, lo: float = -0.25, hi: float = 0.25) -> float:
    return minmax_scale(spread, lo, hi, invert=False)


def score_momentum(ret: float, lo: float = -0.30, hi: float = 0.30) -> float:
    return minmax_scale(ret, lo, hi, invert=False)


def score_volume_trend(ratio: float) -> float:
    return minmax_scale(ratio, 0.5, 2.0, invert=False)


def score_relative_strength(rs: float) -> float:
    return minmax_scale(rs, -0.20, 0.20, invert=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def compute_sentiment_score(
    stock_prices: pd.DataFrame,
    bench_prices: pd.DataFrame = None,
) -> dict:
    """
    Parameters
    ----------
    stock_prices  : DataFrame with 'close' and 'volume' columns
    bench_prices  : DataFrame with 'close' column (Nifty)

    Returns
    -------
    dict with score, sub_scores, metrics, commentary
    """
    if stock_prices is None or stock_prices.empty:
        logger.warning("No stock prices for sentiment model")
        return {"score": 5.0, "sub_scores": {}, "metrics": {}, "commentary": "No data."}

    close  = stock_prices["close"].dropna()
    volume = stock_prices["volume"].dropna() if "volume" in stock_prices.columns else None
    b_close = bench_prices["close"].dropna() if (bench_prices is not None and not bench_prices.empty) else None

    # ── Raw metrics ──────────────────────────────────────────────────────────
    ma50_spread   = _ma_spread(close, 50)
    ma200_spread  = _ma_spread(close, 200)
    mom_1m        = _ret(close, 21)
    mom_3m        = _ret(close, 63)
    mom_6m        = _ret(close, 126)
    rs_3m         = _relative_strength(close, b_close, 63) if b_close is not None else 0.0
    vol_trend     = _volume_trend(volume) if volume is not None else 1.0

    metrics = {
        "ma50_spread":  round(ma50_spread,  4),
        "ma200_spread": round(ma200_spread, 4),
        "mom_1m":       round(mom_1m,  4),
        "mom_3m":       round(mom_3m,  4),
        "mom_6m":       round(mom_6m,  4),
        "rs_vs_nifty":  round(rs_3m,   4),
        "volume_trend": round(vol_trend,4),
    }

    # ── Sub-scores ────────────────────────────────────────────────────────────
    sub = {
        "ma50_score":   score_ma_spread(ma50_spread,  -0.20, 0.20),
        "ma200_score":  score_ma_spread(ma200_spread, -0.30, 0.30),
        "mom1m_score":  score_momentum(mom_1m,  -0.20, 0.20),
        "mom3m_score":  score_momentum(mom_3m,  -0.30, 0.30),
        "mom6m_score":  score_momentum(mom_6m,  -0.45, 0.45),
        "rs_score":     score_relative_strength(rs_3m),
        "vol_score":    score_volume_trend(vol_trend),
    }

    weights = {
        "ma50_score":  0.15,
        "ma200_score": 0.15,
        "mom1m_score": 0.10,
        "mom3m_score": 0.20,
        "mom6m_score": 0.15,
        "rs_score":    0.20,
        "vol_score":   0.05,
    }

    score = float(np.average(list(sub.values()), weights=[weights[k] for k in sub]))
    score = round(score, 4)

    # ── Golden/Death cross flag ───────────────────────────────────────────────
    if len(close) >= 200:
        ma50_val  = close.rolling(50).mean().iloc[-1]
        ma200_val = close.rolling(200).mean().iloc[-1]
        cross_flag = "GOLDEN CROSS ✅" if ma50_val > ma200_val else "DEATH CROSS ⚠️"
    else:
        cross_flag = "Insufficient data"

    commentary = (
        f"MA50 spread: {ma50_spread:+.1%} | MA200 spread: {ma200_spread:+.1%} | "
        f"1M: {mom_1m:+.1%} | 3M: {mom_3m:+.1%} | 6M: {mom_6m:+.1%} | "
        f"RS vs Nifty: {rs_3m:+.1%} | {cross_flag}"
    )

    logger.info("Sentiment score: %.2f | metrics=%s", score, metrics)
    return {
        "score":        score,
        "sub_scores":   sub,
        "metrics":      metrics,
        "commentary":   commentary,
        "cross_flag":   cross_flag,
    }
