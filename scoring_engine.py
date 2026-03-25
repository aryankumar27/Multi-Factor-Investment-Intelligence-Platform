"""
MIIP — scoring_engine.py
Orchestrates all four factor models and produces the final recommendation.
"""

import logging
from datetime import date

import config
from database import save_results, initialise_db
from data_fetcher import fetch_price_data, fetch_benchmark_data, fetch_fundamental_data, fetch_macro_data
from macro_model import compute_macro_score
from fundamental_model import compute_fundamental_score
from risk_model import compute_risk_score
from sentiment_model import compute_sentiment_score

logger = logging.getLogger(__name__)


def recommend(score: float) -> str:
    if score >= config.THRESHOLDS["BUY"]:
        return "BUY"
    elif score >= config.THRESHOLDS["HOLD"]:
        return "HOLD"
    else:
        return "SELL"


def run_analysis(ticker: str) -> dict:
    """
    Full pipeline for a given NSE ticker.
    Returns a rich result dict suitable for the dashboard.
    """
    initialise_db()
    ticker = ticker.upper().strip()
    logger.info("=" * 60)
    logger.info("MIIP Analysis: %s", ticker)
    logger.info("=" * 60)

    # ── Data fetch ─────────────────────────────────────────────────────────
    stock_df, stock_ok   = fetch_price_data(ticker)
    bench_df, bench_ok   = fetch_benchmark_data()
    macro_df, macro_ok   = fetch_macro_data()
    fund_data, fund_ok   = fetch_fundamental_data(ticker)

    data_status = {
        "prices":       stock_ok,
        "benchmark":    bench_ok,
        "macro":        macro_ok,
        "fundamentals": fund_ok,
    }
    logger.info("Data fetch status: %s", data_status)

    # ── Factor models ──────────────────────────────────────────────────────
    macro_result = compute_macro_score(macro_df if macro_ok else None)
    fund_result  = compute_fundamental_score(fund_data)
    risk_result  = compute_risk_score(
        stock_df if stock_ok else None,
        bench_df if bench_ok else None,
    )
    sent_result  = compute_sentiment_score(
        stock_df if stock_ok else None,
        bench_df if bench_ok else None,
    )

    # ── Weighted composite ─────────────────────────────────────────────────
    w = config.WEIGHTS
    final_score = (
        macro_result["score"]  * w["macro"]       +
        fund_result["score"]   * w["fundamental"]  +
        risk_result["score"]   * w["risk"]         +
        sent_result["score"]   * w["sentiment"]
    )
    final_score   = round(final_score, 4)
    recommendation = recommend(final_score)

    logger.info("Final score: %.2f → %s", final_score, recommendation)

    # ── Assemble result ────────────────────────────────────────────────────
    result = {
        "ticker":              ticker,
        "run_date":            str(date.today()),
        "macro_score":         macro_result["score"],
        "fundamental_score":   fund_result["score"],
        "risk_score":          risk_result["score"],
        "sentiment_score":     sent_result["score"],
        "final_score":         final_score,
        "recommendation":      recommendation,
        # detail dicts
        "macro_detail":        macro_result,
        "fundamental_detail":  fund_result,
        "risk_detail":         risk_result,
        "sentiment_detail":    sent_result,
        # price data for charts
        "price_df":            stock_df,
        "bench_df":            bench_df,
        "macro_df":            macro_df,
        # status
        "data_status":         data_status,
        "weights_used":        w,
    }

    # ── Persist ────────────────────────────────────────────────────────────
    save_results(result)
    return result
