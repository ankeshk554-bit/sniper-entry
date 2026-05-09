import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ── 1. TERMINAL CONFIG ──────────────────────────────────────────
st.set_page_config(page_title="Sniper Terminal v3.5", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@800&display=swap');
:root { --bg: #06070a; --accent: #2962ff; --green: #00e676; --red: #ff3d5a; }
html, body, [class*="css"] { background-color: var(--bg) !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetric"] { background: #0e1117 !important; border: 1px solid #1f2436 !important; border-radius: 8px !important; }
.verdict { padding: 12px; border-radius: 4px; text-align: center; font-weight: 800; border: 1px solid; margin-bottom: 20px; }
.bull-v { background: rgba(0,230,118,0.1); color: #00e676; border-color: #00e676; }
.wait-v { background: rgba(209,212,220,0.05); color: #d1d4dc; border-color: #363c4e; }
</style>
""", unsafe_allow_html=True)

# ── 2. ADVANCED DIVERGENCE ENGINE ───────────────────────────────
def get_pivots(series, window=5):
    """Finds local peaks and troughs for line drawing."""
    p_highs = []
    p_lows = []
    for i in range(window, len(series) - window):
        chunk = series.iloc[i-window : i+window+1]
        if series.iloc[i] == chunk.max(): p_highs.append(i)
        if series.iloc[i] == chunk.min(): p_lows.append(i)
    return p_highs, p_lows

def calc_div_lines(df, window=5):
    bull_lines, bear_lines = [], []
    # Using Close for divergence comparison
    p_highs, p_lows = get_pivots(df['Close'], window)
    r_highs, r_lows = get_pivots(df['RSI'], window)

    # Bullish Divergence: Price Lower Low, RSI Higher Low
    for k in range(1, len(p_lows)):
        i1, i2 = p_lows[k-1], p_lows[k]
        # Ensure price is actually making a lower low
        if df['Close'].iloc[i2] < df['Close'].iloc[i1] and df['RSI'].iloc[i2] > df['RSI'].iloc[i1]:
            if df['RSI'].iloc[i2] < 45: # Filter for 'Value Area'
                bull_lines.append(((df.index[i1], df['Close'].iloc[i1]), (df.index[i2], df['Close'].iloc[i2]), 'price'))
                bull_lines.append(((df.index[i1], df['RSI'].iloc[i1]), (df.index[i2], df['RSI'].iloc[i2]), 'rsi'))
    return bull_lines, bear_lines

# ── 3. SIDEBAR & DATA ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏹 SNIPER OPS")
    ticker = st.text_input("Active Target", value="HAL.NS").upper()
    tf_opts = {"Daily": "1d", "Weekly": "1wk", "1 Hour": "1h"}
    interval = tf_opts[st.sidebar.selectbox("Timeframe", list(tf_opts.keys()), index=0)]
    risk_amt = st.sidebar.number_input("Unit Risk (₹)", value=2000)

if ticker:
    try:
        df = yf.download(ticker, period="max" if interval in ["1d", "1wk"] else "60d", interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df['EMA200'] = ta.ema(df['Close'], length=200)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df.dropna(subset=['RSI'], inplace=True)
        bull_lines, _ = calc_div_lines(df)

        # AVWAP
        top_idx, bot_idx = df['High'].argmax(), df['Low'].argmin()
        def av(idx):
            tmp = df.iloc[idx:].copy()
            return (tmp['Close'] * tmp['Volume']).cumsum() / tmp['Volume'].cumsum()
        df['AVWAP_H'], df['AVWAP_L'] = av(top_idx), av(bot_idx)

        # ── 4. UI HUD ──────────────────────────────────────────
        curr = df.iloc[-1]
        is_setup = curr['Close'] > curr['EMA200'] and curr['RSI'] < 40
        verdict_cls = "bull-v" if is_setup else "wait-v"
        verdict_txt = "🎯 A++ SNIPER SETUP DETECTED" if is_setup else "MONITORING STRUCTURE"
        st.markdown(f'<div class="verdict {verdict_cls}">{verdict_txt}</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        diff = curr['Close'] - curr['Low']
        qty = int(risk_amt / diff) if diff > 0 else 0
        c1.metric("Price", f"₹{curr['Close']:,.2f}")
        c2.metric("Sniper Qty", f"{qty}")
        c3.metric("Stop Loss", f"₹{curr['Low']:,.2f}")
        c4.metric("RSI", f"{curr['RSI']:.1f}")

        # ── 5. MASTER CHART ────────────────────────────────────
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Price Panel
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='#2962ff', width=1, dash='dot'), name="EMA 200"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_L'], line=dict(color='#00e676', width=1.5), name="Support"), row=1, col=1)
        
        # RSI Panel
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#00d1ff', width=2), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff3d5a", row=2, col=1, opacity=0.3)
        fig.add_hline(y=30, line_dash="dash", line_color="#00e676", row=2, col=1, opacity=0.3)

        # DRAW DIVERGENCE LINES
        for line in bull_lines:
            # line format: ((x1, y1), (x2, y2), type)
            target_row = 1 if line[2] == 'price' else 2
            fig.add_trace(go.Scatter(
                x=[line[0][0], line[1][0]], y=[line[0][1], line[1][1]],
                mode='lines+markers', line=dict(color='#00e676', width=3),
                marker=dict(size=6), showlegend=False
            ), row=target_row, col=1)

        fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, margin=dict(l=0, r=50, b=0, t=10), paper_bgcolor="#06070a", plot_bgcolor="#06070a")
        fig.update_xaxes(range=[df.index[-100], df.index[-1]], gridcolor="#13161f")
        fig.update_yaxes(side="right", gridcolor="#13161f")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    except Exception as e:
        st.info("Structure mapping in progress...")
