import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- 1. IPAD ULTRA-CONTRAST UI ---
st.set_page_config(page_title="Sniper Terminal v5.1", layout="wide")

st.markdown("""
    <style>
    .main { background: #000000; } /* Pure Black for maximum contrast */
    [data-testid="stSidebar"] { background-color: #111111; border-right: 2px solid #333333; }
    
    /* High-Visibility HUD */
    .hud-card {
        background: #111111;
        border: 2px solid #444444;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .hud-label { color: #AAAAAA; font-size: 14px; font-weight: 700; text-transform: uppercase; }
    .hud-value { color: #00FFCC; font-size: 28px; font-weight: 900; }
    
    .status-banner {
        padding: 15px;
        border-radius: 8px;
        font-size: 20px;
        font-weight: 900;
        text-align: center;
        margin-bottom: 20px;
        border: 2px solid;
    }
    .active { background: rgba(0, 255, 0, 0.2); color: #00FF00; border-color: #00FF00; }
    .neutral { background: rgba(255, 255, 255, 0.1); color: #FFFFFF; border-color: #FFFFFF; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "TITAN.NS"]

st.sidebar.title("🏹 SNIPER OPS")
risk_amt = st.sidebar.number_input("Risk Amount (₹)", value=2000)
tf_options = {"1 Day": "1d", "1 Week": "1wk", "1 Hour": "1h", "15 Min": "15m"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_options.keys()), index=0)
interval = tf_options[selected_tf]

# Watchlist/Search
st.sidebar.markdown("---")
ticker_search = st.sidebar.text_input("🔍 Search Any Stock", "").upper()
ticker = ticker_search if ticker_search else st.sidebar.selectbox("Watchlist", st.session_state.watchlist)

if st.sidebar.button("➕ Add to Watchlist"):
    if ticker_search and ticker_search not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker_search)
        st.rerun()

# --- 3. ROBUST CALCULATIONS ---
def fetch_and_fix(ticker, interval):
    period = "max" if interval in ['1d', '1wk'] else "60d"
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty: return None
    
    # THE NUCLEAR FIX: Flatten any MultiIndex and force standard names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # Technicals
    df['EMA200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # AVWAP Fix: Ensure anchors exist
    high_idx = df['High'].idxmax()
    low_idx = df['Low'].idxmin()
    
    def calc_avwap(anchor_date):
        temp = df.loc[anchor_date:].copy()
        return (temp['Close'] * temp['Volume']).cumsum() / temp['Volume'].cumsum()

    df['AVWAP_TOP'] = calc_avwap(high_idx)
    df['AVWAP_BOT'] = calc_avwap(low_idx)
    
    return df

# --- 4. THE TERMINAL VIEW ---
if ticker:
    df = fetch_and_fix(ticker, interval)
    if df is not None:
        curr = df.iloc[-1]
        
        # Strategy Check
        is_bullish = curr['Close'] > curr['EMA200'] if not pd.isna(curr['EMA200']) else True
        status_class = "active" if is_bullish else "neutral"
        status_text = "BULLISH STRUCTURE" if is_bullish else "BEARISH STRUCTURE"
        st.markdown(f'<div class="status-banner {status_class}">{ticker} | {status_text}</div>', unsafe_allow_html=True)
        
        # Metric Cards
        diff = curr['Close'] - curr['Low']
        qty = int(risk_amt / diff) if diff > 0 else 0
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.markdown(f'<div class="hud-card"><div class="hud-label">Entry</div><div class="hud-value">₹{round(curr["Close"], 1)}</div></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="hud-card"><div class="hud-label">Qty</div><div class="hud-value">{qty}</div></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="hud-card"><div class="hud-label">Stop Loss</div><div class="hud-value">₹{round(curr["Low"], 1)}</div></div>', unsafe_allow_html=True)
        with m4: st.markdown(f'<div class="hud-card"><div class="hud-label">RSI</div><div class="hud-value">{round(curr["RSI"], 1)}</div></div>', unsafe_allow_html=True)

        # --- THE CHART ---
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                            row_width=[0.15, 0.15, 0.7])

        # 1. Price (BOLD NEON)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                                     name="Price", increasing_line_color='#00FF00', decreasing_line_color='#FF0000',
                                     increasing_fillcolor='#00FF00', decreasing_fillcolor='#FF0000'), row=1, col=1)
        
        # Bolder Indicator Lines
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='#00FFFF', width=3, dash='dot'), name="EMA 200"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_TOP'], line=dict(color='#FF00FF', width=4), name="Top Supply"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_BOT'], line=dict(color='#FFFF00', width=4), name="Bot Support"), row=1, col=1)

        # 2. Volume
        v_colors = ['#00FF00' if df['Open'].iloc[i] < df['Close'].iloc[i] else '#FF0000' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=v_colors, name="Volume"), row=2, col=1)

        # 3. RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#FFFFFF', width=3)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#FF0000", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00FF00", opacity=0.5, row=3, col=1)

        # iPad Layout Tweaks
        fig.update_layout(template="plotly_dark", height=900, xaxis_rangeslider_visible=False, 
                          showlegend=False, paper_bgcolor="#000000", plot_bgcolor="#000000",
                          margin=dict(l=10, r=10, b=10, t=10))
        
        # Focus on recent 100 bars
        fig.update_xaxes(range=[df.index[-100], df.index[-1]], showgrid=True, gridcolor="#222222")
        fig.update_yaxes(showgrid=True, gridcolor="#222222", side="right") # Right side labels are easier on iPad
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.error("Engine Fault: Data could not be mapped. Try another ticker.")
