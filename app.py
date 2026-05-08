import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ── 1. GLOBAL TERMINAL CONFIG ───────────────────────────────────
st.set_page_config(
    page_title="Sniper Terminal · Elite v3.0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. ELITE CSS STYLING (Glassmorphism & High-Contrast) ───────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

:root {
    --bg-main: #06070a;
    --bg-card: #0e1117;
    --accent:  #2962ff;
    --green:   #00e676;
    --red:     #ff3d5a;
    --text:    #d1d4dc;
}

html, body, [class*="css"] {
    background-color: var(--bg-main) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Glassmorphism Cards */
[data-testid="stMetric"] {
    background: rgba(14, 17, 23, 0.8) !important;
    border: 1px solid #1f2436 !important;
    border-radius: 8px !important;
}

.verdict-box {
    padding: 15px;
    border-radius: 6px;
    text-align: center;
    font-weight: 800;
    font-size: 1.1rem;
    letter-spacing: 2px;
    margin-bottom: 20px;
    border: 1px solid;
}

.bullish-verdict { background: rgba(0, 230, 118, 0.1); color: var(--green); border-color: var(--green); }
.bearish-verdict { background: rgba(255, 61, 90, 0.1); color: var(--red); border-color: var(--red); }
.neutral-verdict { background: rgba(209, 212, 220, 0.05); color: var(--text); border-color: #363c4e; }

/* Watchlist Chips */
.watchlist-chip {
    background: #1e222d;
    border: 1px solid #363c4e;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 0.7rem;
    color: #00d1ff;
    display: inline-block;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── 3. DATA & ANALYSIS ENGINES ─────────────────────────────────

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "TITAN.NS", "INFY.NS", "HDFCBANK.NS"]

def get_clean_data(symbol, interval):
    period = "max" if interval in ["1d", "1wk"] else "60d"
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_divergence_lines(df, window=5):
    df['Price_Low'] = df['Low'].rolling(window=window*2+1, center=True).min()
    df['Price_High'] = df['High'].rolling(window=window*2+1, center=True).max()
    bull_rsi, bull_price, bear_rsi, bear_price = [], [], [], []

    for i in range(window*2, len(df)-window):
        # Bullish
        if df['Low'].iloc[i] == df['Price_Low'].iloc[i]:
            prev_low_idx = df.iloc[:i-window]['Low'].last_valid_index() # Simplified for logic flow
            # (Detailed pivot logic would follow here - kept compact for v3.0)
    return bull_rsi, bull_price, bear_rsi, bear_price

# ── 4. SIDEBAR COMMANDS ───────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏹 COMMAND CENTER")
    ticker = st.text_input("Active Ticker", value="RELIANCE.NS").upper()
    
    tf_options = {"1W - Weekly": "1wk", "1D - Daily": "1d", "4H - 4 Hour": "4h", "1H - 1 Hour": "1h"}
    interval = tf_options[st.sidebar.selectbox("Analysis Timeframe", list(tf_options.keys()), index=1)]
    
    risk_amt = st.number_input("Risk Capital (₹)", value=2000, step=500)
    
    st.markdown("---")
    st.markdown("### 📂 WATCHLIST")
    chips_html = "".join([f"<span class='watchlist-chip'>{t}</span>" for t in st.session_state.watchlist])
    st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)
    
    new_ticker = st.text_input("Add Ticker", placeholder="e.g. SBIN.NS")
    if st.button("➕ Update Feed"):
        if new_ticker and new_ticker.upper() not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker.upper())
            st.rerun()

# ── 5. MAIN TERMINAL INTERFACE ────────────────────────────────

st.title(f"Institutional Sniper • {ticker}")

if ticker:
    try:
        df = get_clean_data(ticker, interval)
        if df is not None:
            # ── Indicators ──
            df['EMA200'] = ta.ema(df['Close'], length=200)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            # AVWAP Support/Resistance
            top_idx = int(df['High'].argmax())
            bot_idx = int(df['Low'].argmin())
            def av(idx):
                tmp = df.iloc[idx:].copy()
                return (tmp['Close'] * tmp['Volume']).cumsum() / tmp['Volume'].cumsum()
            df['AVWAP_TOP'] = av(top_idx)
            df['AVWAP_BOT'] = av(bot_idx)
            
            curr = df.iloc[-1]
            
            # ── VERDICT HUD ──
            is_uptrend = curr['Close'] > curr['EMA200']
            is_oversold = curr['RSI'] < 35
            
            if is_uptrend and is_oversold:
                st.markdown('<div class="verdict-box bullish-verdict">🎯 A++ SNIPER SETUP: TRENDING & OVERSOLD</div>', unsafe_allow_html=True)
            elif is_uptrend:
                st.markdown('<div class="verdict-box neutral-verdict">STRENGTH: INSTITUTIONAL UPTREND HOLDING</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict-box bearish-verdict">CAUTION: BELOW INSTITUTIONAL FLOOR</div>', unsafe_allow_html=True)

            # ── METRICS ──
            entry, sl = float(curr['Close']), float(curr['Low'])
            diff = entry - sl
            qty = int(risk_amt / diff) if diff > 0 else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Live Price", f"₹{entry:,.2f}")
            c2.metric("Sniper Qty", f"{qty} units")
            c3.metric("Stop Loss", f"₹{sl:,.2f}", delta=f"-{round(diff,2)}", delta_color="inverse")
            c4.metric("Target (1:2)", f"₹{entry + (diff*2):,.2f}")

            # ── WORLD CLASS CHART ──
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                                row_heights=[0.6, 0.15, 0.25])
            
            # Price + EMA + AVWAP
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='#2962ff', width=1.5, dash='dot'), name="EMA 200"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_TOP'], line=dict(color='#ff3d5a', width=1.5), name="Supply"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_BOT'], line=dict(color='#00e676', width=1.5), name="Support"), row=1, col=1)
            
            # Volume
            v_colors = ["#00e676" if df['Open'].iloc[i] < df['Close'].iloc[i] else "#ff3d5a" for i in range(len(df))]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=v_colors, name="Volume", opacity=0.5), row=2, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#00b8d9', width=2), name="RSI"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#ff3d5a", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#00e676", row=3, col=1)

            fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, 
                              margin=dict(l=0, r=50, b=0, t=10), paper_bgcolor="#06070a", plot_bgcolor="#06070a")
            fig.update_xaxes(range=[df.index[-120], df.index[-1]], gridcolor="#13161f")
            fig.update_yaxes(side="right", gridcolor="#13161f")
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    except Exception as e:
        st.error(f"Engine Warning: {e}")
