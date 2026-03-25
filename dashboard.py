"""
MIIP — dashboard.py
Streamlit UI for the Multi-Factor Investment Intelligence Platform
Run: streamlit run dashboard.py
"""

import json
import time
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from scoring_engine import run_analysis
from database import load_history

logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MIIP · Equity Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Metric tiles */
  .metric-tile {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    margin: 0.25rem 0;
  }
  .metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.3rem;
  }
  .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    font-weight: 600;
    color: #f0f6fc;
  }
  .metric-sub {
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 0.2rem;
  }

  /* Recommendation badge */
  .rec-buy  { background:#0d4a20; color:#3fb950; border:1px solid #238636; }
  .rec-hold { background:#3d2c00; color:#d29922; border:1px solid #9e6a03; }
  .rec-sell { background:#4a0000; color:#f85149; border:1px solid #da3633; }
  .rec-badge {
    display:inline-block;
    font-family:'IBM Plex Mono',monospace;
    font-size:2.2rem;
    font-weight:700;
    letter-spacing:0.06em;
    padding:0.5rem 2.5rem;
    border-radius:6px;
    text-align:center;
    margin: 0.6rem 0;
  }

  /* Section headers */
  .section-header {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #388bfd;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #010409;
    border-right: 1px solid #21262d;
  }

  /* Score bar */
  .score-bar-bg {
    background:#21262d; border-radius:4px; height:8px; width:100%; margin-top:6px;
  }
  .score-bar-fill {
    height:8px; border-radius:4px;
    background: linear-gradient(90deg, #388bfd, #3fb950);
  }

  hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────
CHART_BG   = "#0d1117"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"
ACCENT     = "#388bfd"
GREEN      = "#3fb950"
RED        = "#f85149"
YELLOW     = "#d29922"

def _rec_color(rec: str) -> str:
    return {"BUY": GREEN, "HOLD": YELLOW, "SELL": RED}.get(rec, TEXT_COLOR)

def _score_color(s: float) -> str:
    if s >= 7: return GREEN
    if s >= 5: return YELLOW
    return RED


# ─────────────────────────────────────────────────────────────────────────────
# Plotly theme helper
# ─────────────────────────────────────────────────────────────────────────────
def _base_layout(**kwargs) -> dict:
    return dict(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="IBM Plex Mono", color=TEXT_COLOR, size=11),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, showgrid=True),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, showgrid=True),
        margin=dict(l=40, r=20, t=40, b=40),
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chart: Candlestick + MAs + Volume
# ─────────────────────────────────────────────────────────────────────────────
def price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    df = df.copy().tail(252)  # last 12 months

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        name="OHLC", showlegend=False,
    ), row=1, col=1)

    # MAs
    for w, col in [(50, ACCENT), (200, YELLOW)]:
        ma = df["close"].rolling(w).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ma, mode="lines",
            line=dict(color=col, width=1.2, dash="dot"),
            name=f"MA{w}", opacity=0.8,
        ), row=1, col=1)

    # Volume
    colors = [GREEN if df["close"].iloc[i] >= df["open"].iloc[i] else RED
              for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], marker_color=colors,
        opacity=0.5, name="Volume", showlegend=False,
    ), row=2, col=1)

    layout = _base_layout(
        title=dict(text=f"{ticker}  ·  Price & Volume (1Y)", font_size=13, x=0.01),
        legend=dict(bgcolor="rgba(0,0,0,0)", x=0.01, y=0.97),
        xaxis_rangeslider_visible=False,
        height=480,
    )
    layout["yaxis2"] = dict(gridcolor=GRID_COLOR, zeroline=False, showgrid=False)
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart: Radar / Spider
# ─────────────────────────────────────────────────────────────────────────────
def radar_chart(scores: dict) -> go.Figure:
    cats   = list(scores.keys())
    vals   = list(scores.values())
    cats  += [cats[0]]
    vals  += [vals[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats,
        fill="toself",
        fillcolor=f"rgba(56,139,253,0.15)",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=6, color=ACCENT),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=CHART_BG,
            radialaxis=dict(visible=True, range=[0, 10],
                            gridcolor=GRID_COLOR, tickfont_color=TEXT_COLOR,
                            tickfont_size=9),
            angularaxis=dict(gridcolor=GRID_COLOR, tickfont_color=TEXT_COLOR),
        ),
        paper_bgcolor=CHART_BG,
        font=dict(family="IBM Plex Mono", color=TEXT_COLOR),
        height=320,
        margin=dict(l=50, r=50, t=30, b=30),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart: Gauge
# ─────────────────────────────────────────────────────────────────────────────
def gauge_chart(score: float, title: str = "Final Score") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(family="IBM Plex Mono", color=TEXT_COLOR, size=36)),
        title=dict(text=title, font=dict(family="IBM Plex Mono", color=TEXT_COLOR, size=12)),
        gauge=dict(
            axis=dict(range=[0, 10], tickcolor=TEXT_COLOR, tickfont_color=TEXT_COLOR),
            bar=dict(color=_score_color(score), thickness=0.3),
            bgcolor=CHART_BG,
            borderwidth=0,
            steps=[
                dict(range=[0, 5],  color="#1a0000"),
                dict(range=[5, 7],  color="#1a1500"),
                dict(range=[7, 10], color="#001a0a"),
            ],
            threshold=dict(
                line=dict(color="#ffffff", width=2),
                thickness=0.8,
                value=score,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=CHART_BG,
        height=240,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart: Macro series
# ─────────────────────────────────────────────────────────────────────────────
def macro_chart(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return go.Figure()
    df = df.tail(252)
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["NIFTY 50", "USD/INR", "Crude (WTI)", "India VIX"],
        shared_xaxes=False, vertical_spacing=0.18, horizontal_spacing=0.1,
    )
    series_map = {
        (1,1): ("nifty",     GREEN),
        (1,2): ("usdinr",    YELLOW),
        (2,1): ("crude",     ACCENT),
        (2,2): ("india_vix", RED),
    }
    for (r,c), (col, clr) in series_map.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode="lines",
                line=dict(color=clr, width=1.5),
                name=col, showlegend=False,
            ), row=r, col=c)

    fig.update_layout(
        **_base_layout(height=380, title=dict(text="Macro Indicators", font_size=13, x=0.01)),
    )
    for ann in fig.layout.annotations:
        ann.font.color = TEXT_COLOR
        ann.font.size  = 10
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart: Sub-score bar chart
# ─────────────────────────────────────────────────────────────────────────────
def sub_score_chart(sub_scores: dict, title: str) -> go.Figure:
    labels = list(sub_scores.keys())
    values = list(sub_scores.values())
    colors = [_score_color(v) for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors, opacity=0.85,
        text=[f"{v:.1f}" for v in values],
        textposition="outside", textfont=dict(size=9, color=TEXT_COLOR),
    ))
    fig.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="IBM Plex Mono", color=TEXT_COLOR, size=11),
        xaxis=dict(range=[0, 11], gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, showgrid=False),
        margin=dict(l=40, r=20, t=40, b=40),
        height=max(180, len(labels) * 38),
        title=dict(text=title, font_size=12, x=0.01),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def sidebar():
    st.sidebar.markdown("""
    <div style='padding:1rem 0 0.5rem 0;'>
      <div style='font-family:IBM Plex Mono;font-size:1.1rem;font-weight:700;
                  color:#388bfd;letter-spacing:0.08em;'>MIIP</div>
      <div style='font-size:0.7rem;color:#8b949e;letter-spacing:0.06em;
                  margin-top:2px;'>MULTI-FACTOR INVESTMENT INTELLIGENCE PLATFORM</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    ticker = st.sidebar.text_input(
        "NSE Ticker",
        value="RELIANCE",
        placeholder="e.g. TCS, INFY, HDFCBANK",
    ).upper().strip()

    st.sidebar.markdown("<div class='section-header'>Factor Weights</div>", unsafe_allow_html=True)
    macro_w = st.sidebar.slider("Macro",        0.0, 1.0, 0.25, 0.05)
    fund_w  = st.sidebar.slider("Fundamental",  0.0, 1.0, 0.30, 0.05)
    risk_w  = st.sidebar.slider("Risk",         0.0, 1.0, 0.25, 0.05)
    sent_w  = st.sidebar.slider("Sentiment",    0.0, 1.0, 0.20, 0.05)
    total_w = macro_w + fund_w + risk_w + sent_w

    if abs(total_w - 1.0) > 0.01:
        st.sidebar.warning(f"Weights sum to {total_w:.2f} — will be normalised.")
        macro_w, fund_w, risk_w, sent_w = [x/total_w for x in [macro_w, fund_w, risk_w, sent_w]]

    custom_weights = {
        "macro":       macro_w,
        "fundamental": fund_w,
        "risk":        risk_w,
        "sentiment":   sent_w,
    }

    run_btn = st.sidebar.button("▶ Run Analysis", use_container_width=True, type="primary")

    st.sidebar.markdown("""
    <hr>
    <div style='font-size:0.65rem;color:#8b949e;line-height:1.6;'>
    <b>Data sources:</b> Yahoo Finance<br>
    <b>Benchmark:</b> NIFTY 50 (^NSEI)<br>
    <b>DB:</b> SQLite (local)<br>
    <br>
    <b>Score → Signal:</b><br>
    🟢 ≥ 7.0 → BUY<br>
    🟡 5.0–7.0 → HOLD<br>
    🔴 &lt; 5.0 → SELL
    </div>
    """, unsafe_allow_html=True)

    return ticker, custom_weights, run_btn


# ─────────────────────────────────────────────────────────────────────────────
# Metric tile renderer
# ─────────────────────────────────────────────────────────────────────────────
def metric_tile(label: str, value, sub: str = "", color: str = "#f0f6fc"):
    st.markdown(f"""
    <div class='metric-tile'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value' style='color:{color};'>{value}</div>
      <div class='metric-sub'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Score bar
# ─────────────────────────────────────────────────────────────────────────────
def score_bar(label: str, score: float):
    pct = score / 10.0 * 100
    color = _score_color(score)
    st.markdown(f"""
    <div style='margin-bottom:0.7rem;'>
      <div style='display:flex;justify-content:space-between;
                  font-size:0.75rem;color:#8b949e;margin-bottom:4px;'>
        <span>{label}</span>
        <span style='color:{color};font-family:IBM Plex Mono;
                     font-weight:600;'>{score:.2f}</span>
      </div>
      <div class='score-bar-bg'>
        <div class='score-bar-fill' style='width:{pct}%;background:{color};'></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ticker, custom_weights, run_btn = sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='margin-bottom:0.5rem;'>
      <span style='font-family:IBM Plex Mono;font-size:0.7rem;
                   color:#8b949e;letter-spacing:0.12em;'>
        EQUITY RESEARCH DESK · QUANTITATIVE ANALYTICS
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────────
    if "result" not in st.session_state:
        st.session_state.result = None

    if run_btn and ticker:
         with st.spinner(f"Running MIIP analysis for **{ticker}** …"):
            t0 = time.time()
            try:
                result = run_analysis(ticker, weights=custom_weights)
                st.session_state.result = result
                elapsed = time.time() - t0
                st.success(f"Analysis complete in {elapsed:.1f}s")
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

    result = st.session_state.result
    if result is None:
        st.markdown("""
        <div style='text-align:center;padding:5rem 2rem;color:#8b949e;'>
          <div style='font-size:3rem;margin-bottom:1rem;'>📊</div>
          <div style='font-size:1.1rem;font-weight:600;color:#c9d1d9;'>
            Enter a ticker and click <b>Run Analysis</b>
          </div>
          <div style='font-size:0.8rem;margin-top:0.5rem;'>
            Works with any NSE-listed equity (e.g. RELIANCE, TCS, INFY, HDFCBANK)
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    R = result
    ticker    = R["ticker"]
    rec       = R["recommendation"]
    rec_cls   = {"BUY": "rec-buy", "HOLD": "rec-hold", "SELL": "rec-sell"}.get(rec, "")
    rec_color = _rec_color(rec)

    # ── Ticker header row ─────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.markdown(f"""
        <div style='padding:0.5rem 0;'>
          <span style='font-family:IBM Plex Mono;font-size:2rem;
                       font-weight:700;color:#f0f6fc;'>{ticker}</span>
          <span style='font-size:0.75rem;color:#8b949e;
                       margin-left:0.8rem;'>NSE  ·  {R['run_date']}</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='rec-badge {rec_cls}'>{rec}</div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-tile' style='text-align:center;'>
          <div class='metric-label'>MIIP Score</div>
          <div class='metric-value' style='color:{rec_color};font-size:2.4rem;'>
            {R['final_score']:.2f}
          </div>
          <div class='metric-sub'>out of 10</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Factor score tiles ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Factor Scores</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    w = R.get("weights_used", {"macro":0.25,"fundamental":0.30,"risk":0.25,"sentiment":0.20})
    factor_data = [
        ("Macro",       R["macro_score"],       f"{w['macro']*100:.0f}%",       R["macro_detail"]["commentary"]),
        ("Fundamental", R["fundamental_score"],  f"{w['fundamental']*100:.0f}%", R["fundamental_detail"]["commentary"]),
        ("Risk",        R["risk_score"],         f"{w['risk']*100:.0f}%",        R["risk_detail"]["commentary"]),
        ("Sentiment",   R["sentiment_score"],    f"{w['sentiment']*100:.0f}%",   R["sentiment_detail"]["commentary"]),
    ]
    for col, (label, score, weight, comm) in zip(cols, factor_data):
        with col:
            metric_tile(label, f"{score:.2f} / 10",
                        sub=f"Weight: {weight}",
                        color=_score_color(score))

    # ── Score bars + Radar ────────────────────────────────────────────────────
    bar_col, radar_col = st.columns([1, 1])
    with bar_col:
        st.markdown("<div class='section-header'>Score Breakdown</div>", unsafe_allow_html=True)
        score_bar("⬛ Macro",       R["macro_score"])
        score_bar("⬛ Fundamental", R["fundamental_score"])
        score_bar("⬛ Risk",        R["risk_score"])
        score_bar("⬛ Sentiment",   R["sentiment_score"])
        score_bar("★ FINAL",       R["final_score"])
    with radar_col:
        st.plotly_chart(radar_chart({
            "Macro":       R["macro_score"],
            "Fundamental": R["fundamental_score"],
            "Risk":        R["risk_score"],
            "Sentiment":   R["sentiment_score"],
        }), use_container_width=True)

    # ── Price Chart ───────────────────────────────────────────────────────────
    if R["price_df"] is not None and not R["price_df"].empty:
        st.markdown("<div class='section-header'>Price Action</div>", unsafe_allow_html=True)
        st.plotly_chart(price_chart(R["price_df"], ticker), use_container_width=True)

        # Key risk metrics under chart
        rm = R["risk_detail"].get("metrics", {})
        if rm:
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1: metric_tile("Volatility (Ann.)", f"{rm.get('annualised_volatility',0):.1%}", color=YELLOW)
            with m2: metric_tile("Beta",             f"{rm.get('beta',0):.2f}", color=TEXT_COLOR)
            with m3: metric_tile("VaR (95% Daily)",  f"{rm.get('var_95_daily',0):.2%}", color=RED)
            with m4: metric_tile("Max Drawdown",     f"{rm.get('max_drawdown',0):.2%}", color=RED)
            with m5: metric_tile("Sharpe Ratio",     f"{rm.get('sharpe_ratio',0):.2f}", color=GREEN)

    st.markdown("---")

    # ── Factor Deep-Dives (tabs) ──────────────────────────────────────────────
    st.markdown("<div class='section-header'>Factor Deep-Dive</div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Fundamental", "📈 Sentiment", "⚠️ Risk", "🌐 Macro"])

    # Fundamental tab
    with tab1:
        fd = R["fundamental_detail"]
        raw = fd.get("raw_data", {})
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.plotly_chart(sub_score_chart(fd["sub_scores"], "Fundamental Sub-Scores"), use_container_width=True)
        with col_b:
            st.markdown("**Raw Metrics**")
            metrics_table = {
                "PE Ratio":        raw.get("pe_ratio"),
                "PB Ratio":        raw.get("pb_ratio"),
                "ROE":             f"{raw.get('roe',0)*100:.1f}%" if raw.get("roe") else "N/A",
                "Debt / Equity":   raw.get("debt_equity"),
                "Profit Margin":   f"{raw.get('profit_margin',0)*100:.1f}%" if raw.get("profit_margin") else "N/A",
                "Revenue Growth":  f"{raw.get('revenue_growth',0)*100:.1f}%" if raw.get("revenue_growth") else "N/A",
                "Dividend Yield":  f"{raw.get('dividend_yield',0)*100:.2f}%" if raw.get("dividend_yield") else "N/A",
                "Market Cap (Cr)": f"₹{raw.get('market_cap',0)/1e7:.0f}Cr" if raw.get("market_cap") else "N/A",
                "EPS":             raw.get("eps"),
            }
            df_table = pd.DataFrame(list(metrics_table.items()), columns=["Metric", "Value"])
            st.dataframe(df_table, use_container_width=True, hide_index=True)
        st.caption(f"**Commentary:** {fd.get('commentary','')}")

    # Sentiment tab
    with tab2:
        sd = R["sentiment_detail"]
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.plotly_chart(sub_score_chart(sd["sub_scores"], "Sentiment Sub-Scores"), use_container_width=True)
        with col_b:
            st.markdown("**Technical Metrics**")
            metrics_table = {k: f"{v:.2%}" if abs(v) < 5 else f"{v:.2f}"
                             for k, v in sd.get("metrics", {}).items()}
            df_table = pd.DataFrame(list(metrics_table.items()), columns=["Metric", "Value"])
            st.dataframe(df_table, use_container_width=True, hide_index=True)
            cross = sd.get("cross_flag", "")
            if cross:
                st.info(f"Moving Average Signal: **{cross}**")
        st.caption(f"**Commentary:** {sd.get('commentary','')}")

    # Risk tab
    with tab3:
        rd = R["risk_detail"]
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.plotly_chart(sub_score_chart(rd["sub_scores"], "Risk Sub-Scores"), use_container_width=True)
        with col_b:
            st.markdown("**Risk Metrics**")
            rm_data = rd.get("metrics", {})
            fmt_map = {
                "annualised_volatility": lambda v: f"{v:.1%}",
                "beta":                  lambda v: f"{v:.2f}",
                "var_95_daily":          lambda v: f"{v:.2%}",
                "max_drawdown":          lambda v: f"{v:.2%}",
                "sharpe_ratio":          lambda v: f"{v:.2f}",
            }
            rm_display = {k: fmt_map.get(k, str)(v) for k, v in rm_data.items()}
            df_table = pd.DataFrame(list(rm_display.items()), columns=["Metric", "Value"])
            st.dataframe(df_table, use_container_width=True, hide_index=True)
        st.caption(f"**Commentary:** {rd.get('commentary','')}")

    # Macro tab
    with tab4:
        md = R["macro_detail"]
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if R["macro_df"] is not None:
                st.plotly_chart(macro_chart(R["macro_df"]), use_container_width=True)
        with col_b:
            st.plotly_chart(sub_score_chart(md["sub_scores"], "Macro Sub-Scores"), use_container_width=True)
        st.caption(f"**Commentary:** {md.get('commentary','')}")

    # ── Historical runs ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>Historical Analysis Runs</div>", unsafe_allow_html=True)
    history = load_history(ticker, limit=10)
    if history:
        hist_df = pd.DataFrame(history)[[
            "run_date", "macro_score", "fundamental_score",
            "risk_score", "sentiment_score", "final_score", "recommendation"
        ]].rename(columns={
            "run_date":          "Date",
            "macro_score":       "Macro",
            "fundamental_score": "Fundamental",
            "risk_score":        "Risk",
            "sentiment_score":   "Sentiment",
            "final_score":       "Score",
            "recommendation":    "Signal",
        })
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

    # ── Data quality flags ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Data Status</div>", unsafe_allow_html=True)
    ds = R.get("data_status", {})
    sc = st.columns(len(ds))
    for col, (key, ok) in zip(sc, ds.items()):
        with col:
            icon = "✅" if ok else "⚠️"
            st.markdown(f"<div class='metric-tile' style='text-align:center;padding:0.6rem;'>"
                        f"<div style='font-size:1.4rem;'>{icon}</div>"
                        f"<div class='metric-label'>{key}</div></div>", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <hr>
    <div style='font-size:0.65rem;color:#8b949e;text-align:center;line-height:1.8;'>
      MIIP v1.0  ·  For educational & research purposes only  ·
      Not investment advice  ·  Data sourced from Yahoo Finance
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
