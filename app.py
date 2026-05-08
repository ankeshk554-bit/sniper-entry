import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="Sniper Terminal v3.1", layout="wide")

# --- INITIALIZE WATCHLIST ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "TITAN.NS"]

# --- SIDEBAR & NAVIGATION ---
st.sidebar.title("🏹 Sniper Hub")
risk_amt = st.sidebar.number_input("Risk per Trade (₹)", value=2000)
tf_options = {"1 Day": "1d", "1 Week": "1wk", "1 Hour": "1h", "15 Min": "15m", "5 Min": "5m"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_options.keys()), index=0)
interval = tf_options[selected_tf]

# Ticker Selection
ticker = st.sidebar.selectbox("Active Ticker", st.session_state.watchlist)
zoom_range = st.sidebar.slider("Chart Zoom (Bars)", 50, 500, 150)

# Add/Reset Watchlist
st.sidebar.markdown("---")
new_ticker = st.sidebar.text_input("Add Ticker").upper()
if st.sidebar.button("➕ Add"):
    if new_ticker and new_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker)
        st.rerun()

# --- THE CORE ENGINE ---
if ticker:
    try:
        # Fetching Data
        period = "2y" if interval in ["1d", "1wk"] else "60d"
        raw_df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if raw_df.empty:
            st.error("No data found. Ensure ticker is correct.")
        else:
            # 1. CRITICAL FIX: Flatten Multi-Index Columns
            df = raw_df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 2. Technical Indicators
            df['EMA200'] = ta.ema(df['Close'], length=200)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            # 3. Volume Profile (Visible Range Math)
            visible_df = df.tail(zoom_range)
            price_min, price_max = visible_df['Low'].min(), visible_df['High'].max()
            bins = np.linspace(price_min, price_max, 30)
            v_profile = visible_df.groupby(pd.cut(visible_df['Close'], bins, include_lowest=True), observed=False)['Volume'].sum()
            bin_centers = [b.mid for b in v_profile.index]

            # --- SUBPLOT CHART CONSTRUCTION ---
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, 
                                row_width=[0.2, 0.2, 0.6],
                                specs=[[{"secondary_y": True}], [{}], [{}]])

            # PRICE & AVWAP
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            
            # EMA 200
            if not df['EMA200'].dropna().empty:
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='orange', width=1, dash='dot'), name="EMA 200"), row=1, col=1)

            # VOLUME PROFILE OVERLAY
            fig.add_trace(go.Bar(y=bin_centers, x=v_profile.values, orientation='h', 
                                 name="Vol Profile", marker_color='rgba(100, 150, 255, 0.15)', 
                                 showlegend=False, hoverinfo='skip'), row=1, col=1, secondary_y=True)

            # VOLUME BARS
            colors = ['green' if df['Open'].iloc[i] < df['Close'].iloc[i] else 'red' for i in range(len(df))]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)

            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name="RSI"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            # CHART COSMETICS
            fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False,
                              margin=dict(l=10, r=10, b=10, t=30), showlegend=False)
            
            # Adjust Visible Range
            fig.update_xaxes(range=[df.index[-zoom_range], df.index[-1]])
            fig.update_yaxes(showticklabels=False, secondary_y=True) # Hide Vol Profile Axis

            st.plotly_chart(fig, use_container_width=True)

            # --- EXECUTION MATH ---
            curr = df.iloc[-1]
            entry, sl = curr['Close'], curr['Low']
            qty = int(risk_amt / (entry - sl)) if entry > sl else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"₹{round(entry, 2)}")
            c2.metric("Sniper Qty", f"{qty} Shares")
            c3.metric("Target (1:2)", f"₹{round(entry + (entry-sl)*2), 2}")

    except Exception as e:
        st.error(f"⚠️ Dashboard Error: {e}")
