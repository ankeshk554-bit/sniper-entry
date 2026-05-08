import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(page_title="Institutional Sniper Pro", layout="wide")

# --- INITIALIZE WATCHLIST ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "TITAN.NS", "NIFTY_50"]

# --- SIDEBAR: SETTINGS ---
st.sidebar.title("🏹 Sniper Hub")
risk_amt = st.sidebar.number_input("Risk per Trade (₹)", value=2000)

# TIMEFRAME SELECTOR
tf_options = {
    "1 Week": "1wk", "1 Day": "1d", "4 Hour": "4h", 
    "1 Hour": "1h", "15 Min": "15m", "5 Min": "5m", "1 Min": "1m"
}
selected_tf = st.sidebar.selectbox("Select Timeframe", list(tf_options.keys()), index=1)
interval = tf_options[selected_tf]

st.sidebar.markdown("---")
st.sidebar.subheader("Watchlist Manager")
new_ticker = st.sidebar.text_input("Add NSE Stock (e.g. SBIN.NS)").upper()
if st.sidebar.button("➕ Add"):
    if new_ticker and new_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker)
        st.rerun()

if st.sidebar.button("🗑️ Reset"):
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS"]
    st.rerun()

# --- MAIN DASHBOARD: SEARCH ---
st.title("Institutional Sniper Dashboard")
search = st.text_input("🔍 Universal Search", "").upper()
ticker = search if search else st.selectbox("Or select from Watchlist", st.session_state.watchlist)

# --- AVWAP CALCULATION FUNCTION ---
def get_avwap(df, anchor_idx):
    if anchor_idx < 0: return pd.Series(index=df.index)
    temp_df = df.iloc[anchor_idx:].copy()
    pv = (temp_df['Close'] * temp_df['Volume']).cumsum()
    v = temp_df['Volume'].cumsum()
    avwap_series = pv / v
    # Re-index to match original DF length
    return pd.Series(avwap_series, index=df.index).ffill()

if ticker:
    try:
        # Dynamic Period selection to prevent yfinance errors on small intervals
        data_period = "2y" if interval in ["1d", "1wk"] else "60d" if interval in ["1h", "4h", "15m"] else "7d"
        
        with st.spinner(f'Fetching {ticker} ({selected_tf})...'):
            raw_data = yf.download(ticker, period=data_period, interval=interval, progress=False)
            
            if raw_data.empty:
                st.error("⚠️ No data. Add .NS for NSE stocks.")
            else:
                df = raw_data.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # 1. Technicals
                df['EMA200'] = ta.ema(df['Close'], length=200)
                stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
                df = pd.concat([df, stoch], axis=1)
                
                # 2. Find Anchors (Highest High and Lowest Low in View)
                top_idx = df['High'].argmax()
                bot_idx = df['Low'].argmin()
                
                # 3. Calculate AVWAPs
                df['AVWAP_TOP'] = get_avwap(df, top_idx)
                df['AVWAP_BOT'] = get_avwap(df, bot_idx)
                
                curr = df.iloc[-1]
                prev = df.iloc[-2]

                # --- INTERACTIVE CHART ---
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
                
                # Overlays
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='white', width=1, dash='dot'), name="EMA 200"))
                fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_TOP'], line=dict(color='red', width=2), name="AVWAP (Top Supply)"))
                fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_BOT'], line=dict(color='green', width=2), name="AVWAP (Bot Support)"))
                
                fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, b=10, t=40))
                st.plotly_chart(fig, use_container_width=True)

                # --- SNIPER ANALYSIS ---
                is_uptrend = curr['Close'] > curr['EMA200']
                is_oversold = curr['STOCHk_14_3_3'] < 25
                is_trigger = curr['STOCHk_14_3_3'] > curr['STOCHd_14_3_3'] and prev['STOCHk_14_3_3'] <= prev['STOCHd_14_3_3']

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Market Structure")
                    # Analysis against AVWAP
                    if curr['Close'] > curr['AVWAP_TOP']:
                        st.success("🚀 BULLISH: Price cleared Top Supply Line.")
                    elif curr['Close'] < curr['AVWAP_BOT']:
                        st.error("⚠️ BEARISH: Institutional Floor has cracked.")
                    else:
                        st.info("🟡 NEUTRAL: Trading between Top and Bottom AVWAP.")
                    
                    if is_uptrend and is_oversold and is_trigger:
                        st.balloons()
                        st.header("🎯 A++ SIGNAL")

                with col2:
                    st.subheader("Risk Calculator")
                    entry = curr['Close']
                    sl = curr['Low']
                    diff = entry - sl
                    qty = int(risk_amt / diff) if diff > 0 else 0
                    
                    st.write(f"**Stop Loss:** ₹{round(sl, 2)}")
                    st.write(f"**Qty:** {qty} shares (Risk: ₹{risk_amt})")
                    st.write(f"**Target (1:2):** ₹{round(entry + (diff * 2), 2)}")

    except Exception as e:
        st.error(f"Error: {e}")
