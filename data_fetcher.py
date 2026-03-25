"""
MIIP — data_fetcher.py
Retrieves price, macro and fundamental data from Yahoo Finance.
All functions return (data, status_flag) tuples so callers know
whether data is live or from fallback.
"""

import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

import config
from database import save_prices, save_macro, save_fundamentals

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _date_range(years: int) -> Tuple[str, str]:
    end   = datetime.today()
    start = end - timedelta(days=365 * years)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-level columns if present, forward-fill nulls."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.dropna(subset=["close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Price Data
# ─────────────────────────────────────────────────────────────────────────────

def fetch_price_data(ticker: str) -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Download OHLCV history for an NSE ticker.
    Appends .NS if no exchange suffix present.
    Returns (df, success_flag).
    """
    # Normalise ticker
    if "." not in ticker and ticker not in ("^NSEI", "^INDIAVIX", "USDINR=X", "CL=F"):
        yf_ticker = ticker.upper() + ".NS"
    else:
        yf_ticker = ticker.upper()

    start, end = _date_range(config.PRICE_HISTORY_YEARS)
    logger.info("Fetching prices for %s (%s → %s)", yf_ticker, start, end)

    try:
        raw = yf.download(yf_ticker, start=start, end=end, progress=False, auto_adjust=True)
        if raw.empty:
            logger.warning("No price data returned for %s", yf_ticker)
            return None, False
        df = _clean_prices(raw)
        save_prices(ticker.upper(), df)
        return df, True
    except Exception as exc:
        logger.error("Price fetch failed for %s: %s", yf_ticker, exc)
        return None, False


def fetch_benchmark_data() -> Tuple[Optional[pd.DataFrame], bool]:
    return fetch_price_data(config.BENCHMARK_TICKER)


# ─────────────────────────────────────────────────────────────────────────────
# Macro Data
# ─────────────────────────────────────────────────────────────────────────────

def fetch_macro_data() -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Downloads Nifty, USDINR, Crude, India VIX from Yahoo Finance.
    Resamples to monthly, aligns on common dates.
    """
    start, end = _date_range(config.MACRO_HISTORY_YEARS)
    frames = {}
    for name, sym in config.MACRO_TICKERS.items():
        try:
            raw = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
            if raw.empty:
                logger.warning("Macro ticker %s returned no data", sym)
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            s = raw["Close"].squeeze()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            frames[name] = s
        except Exception as exc:
            logger.error("Failed fetching macro %s: %s", sym, exc)

    if not frames:
        return None, False

    df = pd.DataFrame(frames).ffill().dropna()
    df.index.name = "date"
    save_macro(df.reset_index().rename(columns={"index": "date"}))
    logger.info("Macro data fetched: %d rows, %d series", len(df), len(df.columns))
    return df, True


# ─────────────────────────────────────────────────────────────────────────────
# Fundamental Data  (from yfinance .info dict)
# ─────────────────────────────────────────────────────────────────────────────

_FUNDAMENTAL_MAP = {
    "pe_ratio":       ["trailingPE",     "forwardPE"],
    "pb_ratio":       ["priceToBook"],
    "roe":            ["returnOnEquity", "trailingROE", "ReturnOnEquity"],
    "debt_equity":    ["debtToEquity",   "totalDebt"],
    "eps":            ["trailingEps",    "forwardEps"],
    "revenue_growth": ["revenueGrowth",  "earningsGrowth"],
    "profit_margin":  ["profitMargins",  "netMargins"],
    "market_cap":     ["marketCap"],
    "dividend_yield": ["dividendYield",  "trailingAnnualDividendYield"],
}


def _extract(info: dict, keys: list) -> Optional[float]:
    for k in keys:
        v = info.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            # debt_equity from yfinance is already as percentage in some versions
            return float(v)
    return None


def fetch_fundamental_data(ticker: str) -> Tuple[dict, bool]:
    yf_ticker = ticker.upper() + ".NS" if "." not in ticker else ticker.upper()
    logger.info("Fetching fundamentals for %s", yf_ticker)
    try:
        stock = yf.Ticker(yf_ticker)
        info  = stock.info
        data  = {}

        for field, keys in _FUNDAMENTAL_MAP.items():
            val = _extract(info, keys)
            if field == "debt_equity" and val is not None:
                val = val / 100.0
            if field == "roe" and val is not None and abs(val) > 1.5:
                val = val / 100.0
            data[field] = val

        # ── Calculate ROE from financials if yfinance didn't provide it ──
        if data.get("roe") is None:
            try:
                financials    = stock.financials      # income statement
                balance_sheet = stock.balance_sheet   # balance sheet

                # Get most recent Net Income
                net_income = None
                for key in ["Net Income", "NetIncome", "Net Income Common Stockholders"]:
                    if key in financials.index:
                        net_income = float(financials.loc[key].iloc[0])
                        break

                # Get most recent Shareholder Equity
                equity = None
                for key in ["Stockholders Equity", "Total Stockholders Equity",
                            "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                    if key in balance_sheet.index:
                        equity = float(balance_sheet.loc[key].iloc[0])
                        break

                if net_income is not None and equity is not None and equity != 0:
                    data["roe"] = round(net_income / equity, 4)
                    logger.info("ROE calculated from financials: %.4f", data["roe"])
                else:
                    logger.warning("Could not calculate ROE — net_income=%s equity=%s",
                                   net_income, equity)
            except Exception as e:
                logger.warning("ROE calculation from financials failed: %s", e)

        has_data = any(v is not None for v in data.values())
        if has_data:
            save_fundamentals(ticker.upper(), data)
        return data, has_data

    except Exception as exc:
        logger.error("Fundamental fetch failed for %s: %s", yf_ticker, exc)
        return {k: None for k in _FUNDAMENTAL_MAP}, False


