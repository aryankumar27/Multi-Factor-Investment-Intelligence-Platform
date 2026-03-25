"""
MIIP — risk_model.py
Computes the Risk Factor Score (0–10).

Lower risk → higher score.

Metrics:
  • Annualised Volatility
  • Beta vs NIFTY 50
  • Value-at-Risk (95%, parametric)
  • Maximum Drawdown
  • Sharpe Ratio
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from normaliser import minmax_scale
import config

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def _log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def annualised_volatility(prices: pd.Series) -> float:
    r = _log_returns(prices)
    return float(r.std() * np.sqrt(TRADING_DAYS))


def compute_beta(stock_prices: pd.Series, bench_prices: pd.Series) -> float:
    """OLS beta of stock vs benchmark on aligned daily log-returns."""
    stock_ret = _log_returns(stock_prices)
    bench_ret = _log_returns(bench_prices)
    aligned   = pd.concat([stock_ret, bench_ret], axis=1).dropna()
    aligned.columns = ["stock", "bench"]
    if len(aligned) < 30:
        return 1.0
    slope, _, _, _, _ = stats.linregress(aligned["bench"], aligned["stock"])
    return float(slope)


def parametric_var(prices: pd.Series, confidence: float = 0.95) -> float:
    """Daily parametric VaR at given confidence level (negative = loss)."""
    r = _log_returns(prices)
    mu    = r.mean()
    sigma = r.std()
    z     = stats.norm.ppf(1 - confidence)
    return float(mu + z * sigma)      # already negative for losses


def maximum_drawdown(prices: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative value)."""
    roll_max = prices.cummax()
    drawdown = (prices - roll_max) / roll_max
    return float(drawdown.min())


def sharpe_ratio(prices: pd.Series, rfr: float = config.RISK_FREE_RATE) -> float:
    """Annualised Sharpe."""
    r      = _log_returns(prices) * TRADING_DAYS    # scale to annual
    daily  = _log_returns(prices)
    excess = daily - rfr / TRADING_DAYS
    if daily.std() == 0:
        return 0.0
    return float(excess.mean() / daily.std() * np.sqrt(TRADING_DAYS))


# ── Scoring helpers ────────────────────────────────────────────────────────────

def score_volatility(vol: float) -> float:
    """Low vol (0%) → 10;  High vol (100%+) → 0."""
    return minmax_scale(vol, 0.10, 0.80, invert=True)


def score_beta(beta: float) -> float:
    """Beta near 1 is neutral; high beta → riskier → lower score."""
    return minmax_scale(beta, 0.0, 2.5, invert=True)


def score_var(var_95: float) -> float:
    """VaR is negative.  Less negative (→ 0) = lower risk = higher score."""
    return minmax_scale(var_95, -0.08, 0.0, invert=False)


def score_drawdown(mdd: float) -> float:
    """Max drawdown is negative.  Closer to 0 → higher score."""
    return minmax_scale(mdd, -0.70, -0.03, invert=False)


def score_sharpe(sr: float) -> float:
    """Higher Sharpe → better risk-adjusted return → higher score."""
    return minmax_scale(sr, -1.0, 3.0, invert=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def compute_risk_score(
    stock_prices: pd.DataFrame,
    bench_prices: pd.DataFrame,
) -> dict:
    """
    Parameters
    ----------
    stock_prices : pd.DataFrame  (must have 'close' column)
    bench_prices : pd.DataFrame  (Nifty, must have 'close' column)

    Returns
    -------
    dict with score (0-10), sub_scores, metrics, commentary
    """
    if stock_prices is None or stock_prices.empty:
        logger.warning("No stock prices for risk model; returning neutral")
        return {"score": 5.0, "sub_scores": {}, "metrics": {}, "commentary": "No price data."}

    s_close = stock_prices["close"].dropna()
    b_close = bench_prices["close"].dropna() if (bench_prices is not None and not bench_prices.empty) else None

    # ── Raw metrics ───────────────────────────────────────────────────────────
    vol  = annualised_volatility(s_close)
    beta = compute_beta(s_close, b_close) if b_close is not None else 1.0
    var  = parametric_var(s_close)
    mdd  = maximum_drawdown(s_close)
    sr   = sharpe_ratio(s_close)

    metrics = {
        "annualised_volatility": round(vol,  4),
        "beta":                  round(beta, 4),
        "var_95_daily":          round(var,  4),
        "max_drawdown":          round(mdd,  4),
        "sharpe_ratio":          round(sr,   4),
    }

    # ── Sub-scores ────────────────────────────────────────────────────────────
    sub = {
        "volatility_score": score_volatility(vol),
        "beta_score":       score_beta(beta),
        "var_score":        score_var(var),
        "drawdown_score":   score_drawdown(mdd),
        "sharpe_score":     score_sharpe(sr),
    }

    weights = {
        "volatility_score": 0.25,
        "beta_score":       0.20,
        "var_score":        0.20,
        "drawdown_score":   0.20,
        "sharpe_score":     0.15,
    }

    score = float(np.average(list(sub.values()), weights=[weights[k] for k in sub]))
    score = round(score, 4)

    commentary = (
        f"Vol: {vol:.1%} | Beta: {beta:.2f} | "
        f"VaR95: {var:.2%} | MDD: {mdd:.2%} | Sharpe: {sr:.2f}"
    )

    logger.info("Risk score: %.2f | metrics=%s", score, metrics)
    return {
        "score":      score,
        "sub_scores": sub,
        "metrics":    metrics,
        "commentary": commentary,
    }
