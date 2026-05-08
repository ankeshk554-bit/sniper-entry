import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sniper Terminal · Ankesh",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Terminal Aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

:root {
    --bg0:      #08090d;
    --bg1:      #0e1018;
    --bg2:      #13161f;
    --bg3:      #1a1e2a;
    --border:   #1f2436;
    --border2:  #2a3050;
    --text:     #c8cfdf;
    --muted:    #505872;
    --green:    #00e676;
    --red:      #ff3d5a;
    --gold:     #ffc107;
    --cyan:     #00b8d9;
    --accent:   #4f6ef7;
}

html, body, [class*="css"] {
    background-color: var(--bg0) !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 1.8rem !important; max-width: 100% !important; }

[data-testid="stSidebar"] {
    background: var(--bg1) !important;
    border-right: 1px solid var(--border2) !important;
    width: 280px !important;
}

[data-testid="stMetric"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 1rem 1.2rem !important;
}

.watchlist-chip {
    background: var(--bg3);
    border: 1px solid var(--border2);
    border-radius: 4px;
    padding: 0.4rem 0.75rem;
    font-size: 0.72rem;
    color: var(--cyan);
    margin: 2px;
    display: inline-block;
}

.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green);
    display: inline-block;
    box-shadow: 0 0 6px var(--green);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE & SIDEBAR
# ─────────────────────────────────────────────
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "TITAN.NS", "INFY.NS", "HDFCBANK.NS"]

with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.2rem'>
        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:white;letter-spacing:0.04em'>
            ⚡ SNIPER <span style='color:#4f6ef7'>TERMINAL</span>
        </div>
        <div style='font-size:0.6rem;color:#505872;letter-spacing:0.12em;margin-top:2px'>SWING ENGINE v2.1</div>
    </div>
    """, unsafe_allow_html=True)

    ticker = st.text_input("Search Ticker", value="RELIANCE.NS").upper()
    
    tf_options = {"1W — Weekly": "1wk", "1D — Daily": "1d", "4H — 4 Hour": "4h"}
    selected_tf_label = st.selectbox("Timeframe", list(tf_options.keys()), index=1)
    interval = tf_options[selected_tf_label]
    risk_amt = st.number_input("Risk per Trade (₹)", value=2000, step=500)

    st.markdown("### Watchlist")
    chips_html = "".join([f"<span class='watchlist-chip'>{t}</span>" for t in st.session_state.watchlist])
    st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DIVERGENCE CALCULATOR
# ─────────────────────────────────────────────
def calculate_divergence_lines(df, window=5):
    df['Price_Low'] = df['Low'].rolling(window=window*2+1, center=True).min()
    df['Price_High'] = df['High'].rolling(window=window*2+1, center=True).max()
    bull_lines, bear_lines = [], []
    lp_low, lp_high = -1, -1

    for i in range(window*2, len(df)):
        if df['Low'].iloc[i] == df['Price_Low'].iloc[i]:
            if lp_low != -1:
                if df['Low'].iloc[i] < df['Low'].iloc[lp_low] and df['RSI'].iloc[i] > df['RSI'].iloc[lp_low]:
                    if df['RSI'].iloc[i] < 35:
                        bull_lines.append(((df.index[lp_low], df['RSI'].iloc[lp_low]), (df.index[i], df['RSI'].iloc[i])))
                        bull_lines.append(((df.index[lp_low], df['Low'].iloc[lp_low]), (df.index[i], df['Low'].iloc[i]), 'price'))
            lp_low = i
        if df['High'].iloc[i] == df['Price_High'].iloc[i]:
            if lp_high != -1:
                if df['High'].iloc[i] > df['High'].iloc[lp_high] and df['RSI'].iloc[i] < df['RSI'].iloc[lp_high]:
                    if df['RSI'].iloc[i] > 65:
                        bear_lines.append(((df.index[lp_high], df['RSI'].iloc[lp_high]), (df.index[i], df['RSI'].iloc[i])))
                        bear_lines.append(((df.index[lp_high], df['High'].iloc[lp_high]), (df.index[i], df['High'].iloc[i]), 'price'))
            lp_high = i
    return bull_lines, bear_lines

# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────
st.title(f"⚡ Terminal Scan: {ticker}")

try:
    period = "2y" if interval in ["1d", "1wk"] else "60d"
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    
    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df['EMA200'] = ta.ema(df['Close'], length=200)
        df['EMA50']  = ta.ema(df['Close'], length=50)
        df['RSI']    = ta.rsi(df['Close'], length=14)
        bull_lines, bear_lines = calculate_divergence_lines(df)

        # AVWAP Support/Supply
        t_idx, b_idx = df['High'].argmax(), df['Low'].argmin()
        def av(idx):
            tmp = df.iloc[idx:].copy()
            return (tmp['Close'] * tmp['Volume']).cumsum() / tmp['Volume'].cumsum()
        df['AVWAP_TOP'] = av(t_idx)
        df['AVWAP_BOT'] = av(b_idx)

        # Chart Construction
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.25])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color="#4f6ef7", width=1.5, dash="dot"), name="EMA 200"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_TOP'], line=dict(color="#ff3d5a", width=1.5), name="AVWAP Supply"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_BOT'], line=dict(color="#00e676", width=1.5), name="AVWAP Support"), row=1, col=1)
        
        # Volume
        v_colors = ["#00e676" if df['Open'].iloc[i] < df['Close'].iloc[i] else "#ff3d5a" for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=v_colors, name="Volume"), row=2, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color="#00b8d9", width=1.5), name="RSI"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff3d5a", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00e676", row=3, col=1)

        fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, paper_bgcolor="#08090d", plot_bgcolor="#08090d", margin=dict(l=0, r=0, b=0, t=10))
        st.plotly_chart(fig, use_container_width=True)

        # Execution Plan Logic
        curr = df.iloc[-1]
        entry, sl = float(curr['Close']), float(curr['Low'])
        diff = entry - sl
        qty = int(risk_amt / diff) if diff > 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Live Price", f"₹{entry:,.2f}")
        c2.metric("Qty (Risk ₹{risk_amt})", f"{qty}")
        c3.metric("Stop Loss", f"₹{sl:,.2f}", delta=f"-₹{diff:.2f}", delta_color="inverse")
        c4.metric("Target (1:2)", f"₹{entry + (diff*2):,.2f}")

except Exception as e:
    st.error(f"Analysis Error: {e}")
