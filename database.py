"""
MIIP — database.py
Initialises SQLite schema and provides helpers.
"""

import sqlite3
import logging
from config import DB_PATH

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Schema DDL
# ─────────────────────────────────────────────────────────────────────────────
DDL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS prices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    adj_close   REAL,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS macro_data (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    nifty       REAL,
    usdinr      REAL,
    crude       REAL,
    india_vix   REAL,
    UNIQUE(date)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    fetched_at      TEXT    NOT NULL,
    pe_ratio        REAL,
    pb_ratio        REAL,
    roe             REAL,
    debt_equity     REAL,
    eps             REAL,
    revenue_growth  REAL,
    profit_margin   REAL,
    market_cap      REAL,
    dividend_yield  REAL,
    UNIQUE(ticker, fetched_at)
);

CREATE TABLE IF NOT EXISTS results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT    NOT NULL,
    run_date            TEXT    NOT NULL,
    macro_score         REAL,
    fundamental_score   REAL,
    risk_score          REAL,
    sentiment_score     REAL,
    final_score         REAL,
    recommendation      TEXT,
    macro_detail        TEXT,
    fundamental_detail  TEXT,
    risk_detail         TEXT,
    sentiment_detail    TEXT
);

CREATE INDEX IF NOT EXISTS idx_prices_ticker_date   ON prices(ticker, date);
CREATE INDEX IF NOT EXISTS idx_results_ticker       ON results(ticker);
"""


def get_conn() -> sqlite3.Connection:
    """Return a connection with row_factory set."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialise_db() -> None:
    """Create all tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript(DDL)
    logger.info("Database initialised at %s", DB_PATH)


def save_prices(ticker: str, df) -> None:
    """Upsert OHLCV rows into prices table."""
    df = df.copy()
    df["ticker"] = ticker
    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    rows = []
    for _, row in df.iterrows():
        rows.append((
            ticker,
            str(row.get("date", row.get("datetime", "")))[:10],
            row.get("open"),
            row.get("high"),
            row.get("low"),
            row.get("close"),
            row.get("volume"),
            row.get("adj_close", row.get("close")),
        ))

    with get_conn() as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO prices
               (ticker,date,open,high,low,close,volume,adj_close)
               VALUES (?,?,?,?,?,?,?,?)""",
            rows,
        )
    logger.info("Saved %d price rows for %s", len(rows), ticker)


def save_macro(df) -> None:
    """Upsert macro data."""
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append((
            str(row.get("date", ""))[:10],
            row.get("nifty"),
            row.get("usdinr"),
            row.get("crude"),
            row.get("india_vix"),
        ))
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO macro_data (date,nifty,usdinr,crude,india_vix) VALUES (?,?,?,?,?)",
            rows,
        )
    logger.info("Saved %d macro rows", len(rows))


def save_fundamentals(ticker: str, data: dict) -> None:
    """Insert fundamental snapshot."""
    from datetime import date
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO fundamentals
               (ticker,fetched_at,pe_ratio,pb_ratio,roe,debt_equity,eps,
                revenue_growth,profit_margin,market_cap,dividend_yield)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ticker, str(date.today()),
                data.get("pe_ratio"), data.get("pb_ratio"), data.get("roe"),
                data.get("debt_equity"), data.get("eps"),
                data.get("revenue_growth"), data.get("profit_margin"),
                data.get("market_cap"), data.get("dividend_yield"),
            ),
        )


def save_results(record: dict) -> None:
    """Log scoring results."""
    import json
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO results
               (ticker,run_date,macro_score,fundamental_score,risk_score,
                sentiment_score,final_score,recommendation,
                macro_detail,fundamental_detail,risk_detail,sentiment_detail)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record["ticker"], record["run_date"],
                record["macro_score"], record["fundamental_score"],
                record["risk_score"], record["sentiment_score"],
                record["final_score"], record["recommendation"],
                json.dumps(record.get("macro_detail", {})),
                json.dumps(record.get("fundamental_detail", {})),
                json.dumps(record.get("risk_detail", {})),
                json.dumps(record.get("sentiment_detail", {})),
            ),
        )
    logger.info("Results saved for %s  →  %s", record["ticker"], record["recommendation"])


def load_history(ticker: str, limit: int = 10):
    """Return last N result rows for a ticker."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM results WHERE ticker=? ORDER BY run_date DESC LIMIT ?",
            (ticker, limit),
        ).fetchall()
    return [dict(r) for r in rows]
