import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- 1. PREMIUM UI STYLING (THE MODERN LOOK) ---
st.set_page_config(page_title="Sniper Terminal v4.0", layout="wide")

st.markdown("""
    <style>
    /* Global Glassmorphism */
    .main { background: #0E1117; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    
    /* Modern Card Styling */
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 24px; font-weight: 700; color: #00D1FF; }
    .metric-label { font-size: 12px; text-transform: uppercase; color: #8B949E; letter-spacing: 1px; }
    
    /* Custom Status Bar */
    .status-bar {
        padding: 10px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-weight: 600;
        text-align: center;
    }
    .bullish { background: rgba(0, 255, 163, 0.15); color: #00FFA3; border: 1px solid #00FFA3; }
    .bearish { background: rgba(255, 49, 49, 0.15); color: #FF3131; border: 1px solid #FF3131; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIC & DATA ENGINE ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "TITAN.NS"]

# Sidebar - Settings
st.sidebar.title("🏹 Terminal Ops")
risk_amt = st.sidebar.number_input("Risk per Trade (₹)", value=2000)
tf_options = {"Daily": "1d", "Weekly": "1wk", "1 Hour": "1h", "15 Min": "15m"}
selected_tf = st.sidebar.selectbox("Analysis Interval", list(tf_options.keys()), index=0)
interval = tf_options[selected_tf]

# Watchlist Manager
st.sidebar.markdown("---")
new_tick = st.sidebar.text_input("Add Ticker").upper()
if st.sidebar.button("➕ Add to Feed"):
    if new_tick and new_tick not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_tick)
        st.rerun()

ticker = st.sidebar.selectbox("Select Target", st.session_state.watchlist)

# --- 3. DIVERGENCE & AVWAP ENGINE ---
def get_analysis(df):
    df['EMA200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    # AVWAP
    t_idx, b_idx = df['High'].argmax(), df['Low'].argmin()
    def av(idx):
        tmp = df.iloc[idx:].copy()
        return (tmp['Close'] * tmp['Volume']).cumsum() / tmp['Volume'].cumsum()
    df['AVWAP_TOP'] = av(t_idx)
    df['AVWAP_BOT'] = av(b_idx)
    return df

# --- 4. THE UI LAYOUT ---
if ticker:
    data = yf.download(ticker, period="max" if interval in ['1d', '1wk'] else "60d", interval=interval, progress=False, auto_adjust=True)
    if not data.empty:
        df = data.copy()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = get_analysis(df)
        curr = df.iloc[-1]
        
        # --- TOP HUD (HEADS-UP DISPLAY) ---
        is_bullish = curr['Close'] > curr['EMA200']
        status_class = "bullish" if is_bullish else "bearish"
        status_text = "INSTITUTIONAL UPTREND" if is_bullish else "INSTITUTIONAL DOWNTREND"
        
        st.markdown(f'<div class="status-bar {status_class}">{status_text} | {ticker} @ ₹{round(curr["Close"], 2)}</div>', unsafe_allow_html=True)
        
        # Metric Cards
        diff = curr['Close'] - curr['Low']
        qty = int(risk_amt / diff) if diff > 0 else 0
        
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.markdown(f'<div class="metric-card"><div class="metric-label">Position Qty</div><div class="metric-value">{qty}</div></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="metric-card"><div class="metric-label">Stop Loss</div><div class="metric-value">₹{round(curr["Low"], 2)}</div></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="metric-card"><div class="metric-label">Target (1:2)</div><div class="metric-value">₹{round(curr["Close"] + (diff*2), 2)}</div></div>', unsafe_allow_html=True)
        with m4: st.markdown(f'<div class="metric-card"><div class="metric-label">RSI</div><div class="metric-value">{round(curr["RSI"], 1)}</div></div>', unsafe_allow_html=True)

        # --- THE MAIN CHART ---
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, 
                            row_width=[0.2, 0.2, 0.6], subplot_titles=("Price Action", "Volume", "RSI Momentum"))
        
        # Candle + Overlays
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='#00D1FF', width=1, dash='dot'), name="EMA 200"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_TOP'], line=dict(color='#FF3131', width=1.5), name="Supply"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_BOT'], line=dict(color='#00FFA3', width=1.5), name="Support"), row=1, col=1)
        
        # Volume
        colors = ['#00FFA3' if df['Open'].iloc[i] < df['Close'].iloc[i] else '#FF3131' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#ab63fa', width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#FF3131", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00FFA3", row=3, col=1)

        fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=10, r=10, b=10, t=30))
        # Snap to recent action
        fig.update_xaxes(range=[df.index[-120], df.index[-1]])
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Connection lost. Retrying data stream...")
