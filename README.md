# MIIP · Multi-Factor Investment Intelligence Platform
### Automated Equity Research Dashboard — Indian Equities (NSE)

---

## Architecture

```
User Input (Ticker)
        │
        ▼
┌───────────────────┐
│   data_fetcher.py │  ← Yahoo Finance API
│  Price │ Macro    │
│  Fundamental      │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│    database.py    │  ← SQLite (miip.db)
│  prices │ macro   │
│  fundamentals     │
│  results          │
└────────┬──────────┘
         │
    ┌────┴────────────────────────┐
    ▼         ▼          ▼        ▼
macro_  fundamental_  risk_   sentiment_
model     model       model    model
    └────┬────────────────────────┘
         │
         ▼
┌────────────────────┐
│  scoring_engine.py │
│  Weighted composite│
│  Recommendation    │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   dashboard.py     │  ← Streamlit UI
│  Charts │ Tables   │
│  Gauges │ Radar    │
└────────────────────┘
```

---

## Factor Model

| Factor        | Weight | Key Inputs                              |
|---------------|--------|-----------------------------------------|
| Macro         | 25%    | Nifty trend, USDINR, Crude, India VIX   |
| Fundamental   | 30%    | PE, PB, ROE, D/E, Margin, Growth        |
| Risk          | 25%    | Volatility, Beta, VaR, MDD, Sharpe      |
| Sentiment     | 20%    | MA crossovers, Momentum, RS vs Nifty    |

**Recommendation Rules:**
- Score ≥ 7.0 → **BUY**
- Score 5.0–7.0 → **HOLD**
- Score < 5.0 → **SELL**

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the dashboard
```bash
streamlit run dashboard.py
```

### 3. Enter any NSE ticker
Examples: `RELIANCE`, `TCS`, `INFY`, `HDFCBANK`, `SBIN`, `WIPRO`

---

## Module Reference

| File                  | Purpose                                      |
|-----------------------|----------------------------------------------|
| `config.py`           | Weights, DB path, thresholds, bounds         |
| `database.py`         | SQLite schema, read/write helpers            |
| `data_fetcher.py`     | Yahoo Finance data retrieval                 |
| `normaliser.py`       | Min-max scaling, score utilities             |
| `macro_model.py`      | Macro factor scoring                         |
| `fundamental_model.py`| Fundamental factor scoring                   |
| `risk_model.py`       | Risk factor scoring (vol, beta, VaR, MDD)    |
| `sentiment_model.py`  | Technical/sentiment factor scoring           |
| `scoring_engine.py`   | Orchestration & weighted composite           |
| `dashboard.py`        | Streamlit UI                                 |
| `miip.db`             | Auto-created SQLite database                 |

---

## Customisation

Edit `config.py` to change:
- `WEIGHTS` — factor weights (must sum to 1.0)
- `THRESHOLDS` — BUY/SELL thresholds
- `RISK_FREE_RATE` — Indian G-Sec rate
- `PRICE_HISTORY_YEARS` — historical data range
- `NORM_BOUNDS` — scoring bounds per metric

---

## Disclaimer

This platform is for educational and research purposes only.
It does not constitute investment advice.
Always conduct independent due diligence before making investment decisions.
