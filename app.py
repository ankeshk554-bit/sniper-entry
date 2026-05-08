import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="Sniper Terminal v3.2", layout="wide")

# --- INITIALIZE WATCHLIST ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "TITAN.NS"]

# --- SIDEBAR & NAVIGATION ---
st.sidebar.title("🏹 Sniper Hub")
risk_amt = st.sidebar.number_input("Risk per Trade (₹)", value=2000)

# Timeframe logic with enough buffer for EMA 200
tf_options = {"1 Week": "1wk", "1 Day": "1d", "4 Hour": "4h", "1 Hour": "1h", "15 Min": "15m", "5 Min": "5m"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_options.keys()), index=1)
interval = tf_options[selected_tf]

ticker = st.sidebar.selectbox("Active Ticker", st.session_state.watchlist)
zoom_range = st.sidebar.slider("Chart Zoom (Bars)", 50, 500, 150)

# Add Ticker
st.sidebar.markdown("---")
new_ticker = st.sidebar.text_input("Add Ticker").upper()
if st.sidebar.button("➕ Add"):
    if new_ticker and new_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker)
        st.rerun()

# --- THE CORE ENGINE ---
if ticker:
    try:
        # BUFFER FIX: Fetch more data than needed to ensure EMA200 calculates correctly
        # Weekly needs ~5-10 years, Intraday needs maximum available
        data_period = "max" if interval in ["1wk", "1d"] else "60d"
        
        with st.spinner(f'Analyzing {ticker}...'):
            raw_df = yf.download(ticker, period=data_period, interval=interval, progress=False)
        
        if raw_df.empty or len(raw_df) < 20:
            st.error("No data found. Check ticker or timeframe.")
        else:
            df = raw_df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 1. Technical Indicators (Must come BEFORE zoom/slicing)
            df['EMA200'] = ta.ema(df['Close'], length=200)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            # 2. SAFETY FIX: Cap the zoom range to actual data size
            available_bars = len(df)
            safe_zoom = min(zoom_range, available_bars - 1)
            
            # 3. Volume Profile (Calculated ONLY on visible range)
            visible_df = df.tail(safe_zoom)
            price_min, price_max = visible_df['Low'].min(), visible_df['High'].max()
            if price_min != price_max: # Prevent crash on flat data
                bins = np.linspace(price_min, price_max, 30)
                v_profile = visible_df.groupby(pd.cut(visible_df['Close'], bins, include_lowest=True), observed=False)['Volume'].sum()
                bin_centers = [b.mid for b in v_profile.index]
            else:
                bin_centers, v_profile = [], []

            # --- SUBPLOT CHART ---
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, 
                                row_width=[0.2, 0.2, 0.6],
                                specs=[[{"secondary_y": True}], [{}], [{}]])

            # PRICE
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            
            # EMA 200 (Only plot if calculated)
            if 'EMA200' in df and not df['EMA200'].dropna().empty:
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='orange', width=1.5, dash='dot'), name="EMA 200"), row=1, col=1)

            # VOLUME PROFILE
            if len(bin_centers) > 0:
                fig.add_trace(go.Bar(y=bin_centers, x=v_profile.values, orientation='h', 
                                     name="Vol Profile", marker_color='rgba(100, 150, 255, 0.2)', 
                                     showlegend=False, hoverinfo='skip'), row=1, col=1, secondary_y=True)

            # VOLUME BARS
            v_colors = ['green' if df['Open'].iloc[i] < df['Close'].iloc[i] else 'red' for i in range(len(df))]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=v_colors, name="Volume"), row=2, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name="RSI"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            # CHART COSMETICS
            fig.update_layout(template="plotly_dark", height=850, xaxis_rangeslider_visible=False,
                              margin=dict(l=10, r=10, b=10, t=30), showlegend=False)
            
            # DYNAMIC ZOOM FIX: Ensures we don't index out of bounds
            fig.update_xaxes(range=[df.index[-safe_zoom], df.index[-1]])
            fig.update_yaxes(showticklabels=False, secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)

            # --- STATS METRICS ---
            curr = df.iloc[-1]
            entry, sl = curr['Close'], curr['Low']
            qty = int(risk_amt / (entry - sl)) if entry > sl else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Live Price", f"₹{round(entry, 2)}")
            c2.metric("Sniper Qty", f"{qty} Shares")
            c3.metric("EMA 200", f"₹{round(df['EMA200'].iloc[-1], 2)}" if not pd.isna(df['EMA200'].iloc[-1]) else "Calculating...")

    except Exception as e:
        st.error(f"⚠️ App Resetting: {e}")
