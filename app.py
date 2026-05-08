import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(page_title="Ankesh Sniper Pro", layout="wide")

# --- INITIALIZE WATCHLIST ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "TITAN.NS"]

# --- SIDEBAR: SETTINGS & WATCHLIST ---
st.sidebar.title("🏹 Sniper Hub")
risk_amt = st.sidebar.number_input("Risk per Trade (₹)", value=2000)

st.sidebar.markdown("---")
st.sidebar.subheader("Watchlist Manager")
new_ticker = st.sidebar.text_input("Add to Watchlist (e.g. SBIN.NS)").upper()
if st.sidebar.button("➕ Add"):
    if new_ticker and new_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker)
        st.rerun()

if st.sidebar.button("🗑️ Reset List"):
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS"]
    st.rerun()

# --- MAIN DASHBOARD: UNIVERSAL SEARCH ---
st.title("Institutional Sniper Dashboard")
universal_search = st.text_input("🔍 Universal Search (Type any ticker like AAPL or INFY.NS)", "").upper()

# Determine which ticker to analyze
ticker_to_run = universal_search if universal_search else st.selectbox("Or select from Watchlist", st.session_state.watchlist)

if ticker_to_run:
    try:
        with st.spinner(f'Fetching live data for {ticker_to_run}...'):
            # FIX: auto_adjust=True and flattening columns for the blank chart bug
            data = yf.download(ticker_to_run, period="2y", interval="1d", progress=False)
            
            if data.empty or len(data) < 200:
                st.error(f"⚠️ No data found for {ticker_to_run}. Try adding .NS for NSE stocks.")
            else:
                # FIX: Flatten Multi-Index columns from yfinance
                df = data.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Technical Calculations
                df['EMA200'] = ta.ema(df['Close'], length=200)
                df['EMA50'] = ta.ema(df['Close'], length=50)
                stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
                df = pd.concat([df, stoch], axis=1)
                
                curr = df.iloc[-1]
                prev = df.iloc[-2]

                # --- INTERACTIVE CANDLESTICK CHART ---
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='orange', width=2), name="EMA 200 (Institutional Floor)"))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='cyan', width=1), name="EMA 50"))
                
                fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, 
                                  margin=dict(l=10, r=10, b=10, t=40), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)

                # --- SNIPER ANALYSIS ---
                st.markdown("---")
                is_uptrend = curr['Close'] > curr['EMA200']
                is_oversold = curr['STOCHk_14_3_3'] < 25
                is_trigger = curr['STOCHk_14_3_3'] > curr['STOCHd_14_3_3'] and prev['STOCHk_14_3_3'] <= prev['STOCHd_14_3_3']

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Strategy Status")
                    if is_uptrend and is_oversold and is_trigger:
                        st.success("🎯 A++ SETUP DETECTED!")
                        st.balloons()
                    elif is_uptrend and is_oversold:
                        st.info("🟡 In Value Zone: Waiting for Stochastic Crossover.")
                    elif not is_uptrend:
                        st.error("❌ Below Institutional Floor (EMA 200). Avoid Longs.")
                    else:
                        st.warning("⚪ Neutral: Waiting for Pullback to EMA.")

                with col2:
                    st.subheader("Sniper Trade Plan")
                    entry = curr['Close']
                    sl = curr['Low']
                    risk_per_share = entry - sl
                    qty = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
                    
                    if qty > 0:
                        st.write(f"**Stock:** {ticker_to_run}")
                        st.write(f"**Qty to Buy:** {qty} shares")
                        st.write(f"**Stop Loss:** ₹{round(sl, 2)}")
                        st.write(f"**Target (1:2):** ₹{round(entry + (risk_per_share * 2), 2)}")
                        st.caption(f"Calculated for a fixed ₹{risk_amt} risk.")
                    else:
                        st.write("Price is currently at/below SL. No trade possible.")

    except Exception as e:
        st.error(f"System Error: {e}")
