import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- 1. BLOOMBERG DARK THEME UI ---
st.set_page_config(page_title="Sniper Terminal v5.0", layout="wide")

st.markdown("""
    <style>
    .main { background: #0B0E11; font-family: 'SF Pro Display', sans-serif; }
    [data-testid="stSidebar"] { background-color: #15191E; border-right: 1px solid #2B3139; }
    
    /* Premium HUD Cards */
    .hud-card {
        background: #1E222D;
        border-top: 2px solid #2962FF;
        border-radius: 4px;
        padding: 15px;
        text-align: left;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        margin-bottom: 10px;
    }
    .hud-label { color: #848E9C; font-size: 11px; text-transform: uppercase; font-weight: 600; margin-bottom: 5px;}
    .hud-value { color: #F0B90B; font-size: 20px; font-weight: 700; }
    
    /* Sniper Badge */
    .sniper-status {
        padding: 8px 15px;
        border-radius: 4px;
        font-weight: 800;
        text-align: center;
        letter-spacing: 1px;
        margin-bottom: 20px;
    }
    .active { background: #02C076; color: white; box-shadow: 0 0 15px rgba(2, 192, 118, 0.4); }
    .wait { background: #1E222D; color: #848E9C; border: 1px solid #474D57; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE CORE ENGINE ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "TITAN.NS", "NIFTY_50"]

st.sidebar.title("⚓ COMMAND CENTER")
risk_amt = st.sidebar.number_input("Unit Risk (₹)", value=2000, step=100)
tf_map = {"Daily": "1d", "Weekly": "1wk", "4H": "4h", "1H": "1h", "15M": "15m"}
selected_tf = st.sidebar.selectbox("Interval", list(tf_map.keys()), index=0)
interval = tf_map[selected_tf]

# Ticker Manager
st.sidebar.markdown("---")
new_tick = st.sidebar.text_input("➕ ADD TICKER (e.g. AAPL)").upper()
if st.sidebar.button("SYNC FEED"):
    if new_tick and new_tick not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_tick)
        st.rerun()

ticker = st.sidebar.selectbox("ACTIVE TARGET", st.session_state.watchlist)

# --- 3. ROBUST AVWAP & TECHNICALS ---
def calculate_world_class_data(ticker, interval):
    period = "max" if interval in ['1d', '1wk'] else "60d"
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty: return None
    
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Technical Indicators
    df['EMA200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # FIXED AVWAP LOGIC: Anchored to Screen Extremes
    def avwap(anchor_idx):
        tmp = df.iloc[anchor_idx:].copy()
        if tmp['Volume'].sum() == 0: return pd.Series(index=df.index)
        val = (tmp['Close'] * tmp['Volume']).cumsum() / tmp['Volume'].cumsum()
        return val.reindex(df.index).ffill()

    df['AVWAP_H'] = avwap(df['High'].argmax())
    df['AVWAP_L'] = avwap(df['Low'].argmin())
    
    return df

# --- 4. EXECUTION & CHARTING ---
if ticker:
    df = calculate_world_class_data(ticker, interval)
    if df is not None:
        curr = df.iloc[-1]
        
        # --- UI TOP ROW (HUD) ---
        is_uptrend = curr['Close'] > curr['EMA200']
        is_value = curr['RSI'] < 40
        sniper_active = is_uptrend and is_value
        
        status_css = "active" if sniper_active else "wait"
        status_text = "SNIPER READY: A++ SETUP" if sniper_active else "WAITING FOR VALUE"
        st.markdown(f'<div class="sniper-status {status_css}">{status_text}</div>', unsafe_allow_html=True)
        
        # Risk Math
        sl = curr['Low']
        diff = curr['Close'] - sl
        qty = int(risk_amt / diff) if diff > 0 else 0
        
        h1, h2, h3, h4 = st.columns(4)
        with h1: st.markdown(f'<div class="hud-card"><div class="hud-label">Entry Price</div><div class="hud-value">₹{round(curr["Close"], 2)}</div></div>', unsafe_allow_html=True)
        with h2: st.markdown(f'<div class="hud-card"><div class="hud-label">Position Qty</div><div class="hud-value">{qty}</div></div>', unsafe_allow_html=True)
        with h3: st.markdown(f'<div class="hud-card"><div class="hud-label">Stop Loss</div><div class="hud-value">₹{round(sl, 2)}</div></div>', unsafe_allow_html=True)
        with h4: st.markdown(f'<div class="hud-card"><div class="hud-label">RSI Gauge</div><div class="hud-value">{round(curr["RSI"], 1)}</div></div>', unsafe_allow_html=True)

        # --- THE CHART (TradingView Desktop Style) ---
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                            row_width=[0.15, 0.15, 0.7], 
                            subplot_titles=("", "", ""))
        
        # 1. Price + AVWAPs
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                                     name="Price", increasing_line_color='#02C076', decreasing_line_color='#F84960'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='#2962FF', width=1.5, dash='dot'), name="Institutional Floor"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_H'], line=dict(color='#F84960', width=2), name="Upper Supply"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_L'], line=dict(color='#02C076', width=2), name="Lower Support"), row=1, col=1)
        
        # 2. Volume
        v_colors = ['#02C076' if df['Open'].iloc[i] < df['Close'].iloc[i] else '#F84960' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=v_colors, name="Volume"), row=2, col=1)
        
        # 3. RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#F0B90B', width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#F84960", opacity=0.3, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#02C076", opacity=0.3, row=3, col=1)

        # WORLD CLASS LAYOUT TWEAKS
        fig.update_layout(template="plotly_dark", height=850, xaxis_rangeslider_visible=False, showlegend=False, 
                          margin=dict(l=10, r=10, b=10, t=10), paper_bgcolor="#0B0E11", plot_bgcolor="#0B0E11")
        
        fig.update_xaxes(range=[df.index[-120], df.index[-1]], showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#232730", zeroline=False)
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.error("Engine failure: Check ticker connectivity.")
