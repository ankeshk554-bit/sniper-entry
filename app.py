import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="Sniper Terminal v3.0", layout="wide")

# --- UI TABS FOR CLEANER NAVIGATION ---
tab1, tab2 = st.tabs(["🏹 Trading Desk", "📂 Watchlist Manager"])

with tab2:
    st.subheader("Manage Your Tickers")
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "TITAN.NS"]
    
    col_a, col_b = st.columns(2)
    with col_a:
        new_ticker = st.text_input("Add Ticker (e.g. INFIBEAM.NS)").upper()
        if st.button("➕ Add to Watchlist"):
            if new_ticker and new_ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_ticker)
                st.rerun()
    with col_b:
        if st.button("🗑️ Reset to Default"):
            st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "TITAN.NS"]
            st.rerun()

with tab1:
    # --- SIDEBAR CONTROLS ---
    st.sidebar.title("Sniper Controls")
    risk_amt = st.sidebar.number_input("Risk Amount (₹)", value=2000)
    tf_options = {"1 Day": "1d", "1 Week": "1wk", "1 Hour": "1h", "15 Min": "15m", "5 Min": "5m"}
    selected_tf = st.sidebar.selectbox("Timeframe", list(tf_options.keys()), index=0)
    interval = tf_options[selected_tf]
    
    ticker = st.sidebar.selectbox("Active Ticker", st.session_state.watchlist)
    st.sidebar.markdown("---")
    zoom_range = st.sidebar.slider("Chart View (Bars)", 50, 500, 150)

    # --- DATA FETCHING ---
    if ticker:
        try:
            period = "2y" if interval in ["1d", "1wk"] else "60d"
            df_raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            
            if not df_raw.empty:
                df = df_raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Calculations
                df['EMA200'] = ta.ema(df['Close'], length=200)
                df['RSI'] = ta.rsi(df['Close'], length=14)
                
                # AVWAP Support/Resistance
                top_idx, bot_idx = df['High'].argmax(), df['Low'].argmin()
                def calc_avwap(df, idx):
                    temp = df.iloc[idx:].copy()
                    return (temp['Close'] * temp['Volume']).cumsum() / temp['Volume'].cumsum()
                df['AVWAP_TOP'] = calc_avwap(df, top_idx)
                df['AVWAP_BOT'] = calc_avwap(df, bot_idx)

                # --- VOLUME PROFILE CALCULATION ---
                # We calculate volume at price levels for the 'Visible Range'
                visible_df = df.tail(zoom_range)
                price_min, price_max = visible_df['Low'].min(), visible_df['High'].max()
                bins = np.linspace(price_min, price_max, 40)
                v_profile = visible_df.groupby(pd.cut(visible_df['Close'], bins))['Volume'].sum()
                bin_centers = [b.mid for b in v_profile.index]

                # --- CHART CONSTRUCTION ---
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                                    row_width=[0.2, 0.2, 0.6], 
                                    specs=[[{"secondary_y": True}], [{}], [{}]])

                # 1. MAIN PRICE CHART
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='orange', dash='dot'), name="EMA 200"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_TOP'], line=dict(color='red', width=2), name="AVWAP TOP"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_BOT'], line=dict(color='green', width=2), name="AVWAP BOT"), row=1, col=1)

                # 2. VOLUME PROFILE (Overlay on Price)
                fig.add_trace(go.Bar(y=bin_centers, x=v_profile.values, orientation='h', name="Volume Profile",
                                     marker_color='rgba(100, 150, 255, 0.2)', hoverinfo='skip'), row=1, col=1, secondary_y=True)

                # 3. VOLUME BARS
                v_colors = ['green' if df['Open'].iloc[i] < df['Close'].iloc[i] else 'red' for i in range(len(df))]
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=v_colors, name="Volume"), row=2, col=1)

                # 4. RSI
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name="RSI"), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

                # NAVIGATION & LAYOUT
                fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False,
                                  showlegend=False, margin=dict(l=10, r=10, b=10, t=50),
                                  xaxis=dict(range=[df.index[-zoom_range], df.index[-1]]))
                
                # Disable primary chart y-axis scaling for Volume Profile overlay
                fig.update_yaxes(showgrid=False, secondary_y=True, showticklabels=False)
                
                st.plotly_chart(fig, use_container_width=True)

                # --- EXECUTION MATH ---
                curr = df.iloc[-1]
                entry, sl = curr['Close'], curr['Low']
                diff = entry - sl
                qty = int(risk_amt / diff) if diff > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Strategy Signal", "A++ SETUP" if (curr['Close'] > df['EMA200'].iloc[-1] and curr['RSI'] < 35) else "WAITING")
                col2.metric("Buy Quantity", f"{qty} Shares")
                col3.metric("Trade Target (1:2)", f"₹{round(entry + (diff * 2), 2)}")

        except Exception as e:
            st.error(f"Analysis Error: {e}")
