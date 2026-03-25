"""
MIIP — Multi-Factor Investment Intelligence Platform
config.py  |  Global configuration & factor weights
Author: Equity Research Desk
"""

import os

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "miip.db")

# ── Date range for historical data ───────────────────────────────────────────
PRICE_HISTORY_YEARS = 3          # years of OHLCV to pull
MACRO_HISTORY_YEARS = 3

# ── Benchmark ─────────────────────────────────────────────────────────────────
BENCHMARK_TICKER = "^NSEI"       # NIFTY 50

# ── Risk-free rate (annualised, as decimal) ──────────────────────────────────
RISK_FREE_RATE = 0.068           # ~6.8% (current Indian 10Y G-Sec)

# ── Factor weights  (must sum to 1.0) ────────────────────────────────────────
WEIGHTS = {
    "macro":        0.25,
    "fundamental":  0.30,
    "risk":         0.25,
    "sentiment":    0.20,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Factor weights must sum to 1.0"

# ── Score thresholds ─────────────────────────────────────────────────────────
THRESHOLDS = {
    "BUY":  7.0,
    "HOLD": 5.0,
    # below 5.0 → SELL
}

# ── Macro indicator tickers (Yahoo Finance) ───────────────────────────────────
MACRO_TICKERS = {
    "nifty":      "^NSEI",
    "usdinr":     "USDINR=X",
    "crude":      "CL=F",          # WTI crude
    "india_vix":  "^INDIAVIX",
}

# ── FRED series IDs (fallback / supplemental) ─────────────────────────────────
FRED_SERIES = {
    "us_fed_rate": "FEDFUNDS",
    "us_cpi":      "CPIAUCSL",
}

# ── Normalisation bounds (min–max per factor sub-metric) ─────────────────────
# Used to clamp outliers before min-max scaling
NORM_BOUNDS = {
    "pe_ratio":        (0,  80),
    "pb_ratio":        (0,  20),
    "roe":             (-0.5, 0.6),
    "debt_equity":     (0,  5),
    "volatility_ann":  (0,  1.5),
    "beta":            (-1, 3),
    "var_95":          (-0.2, 0),
    "max_drawdown":    (-1,  0),
    "momentum_1m":     (-0.3, 0.3),
    "momentum_3m":     (-0.5, 0.5),
    "momentum_6m":     (-0.7, 0.7),
    "ma_spread":       (-0.3, 0.3),
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
