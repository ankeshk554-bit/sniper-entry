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

# --- SIDEBAR ---
st.sidebar.title("🏹 Sniper Hub")
risk_amt = st.sidebar.number_input("Risk per Trade (₹)", value=2000)

st.sidebar.markdown("---")
st.sidebar.subheader("Watchlist Manager")
new_ticker = st.sidebar.text_input("Add NSE Stock (e.g. SBIN.NS)").upper()
if st.sidebar.button("➕ Add Stock"):
    if new_ticker and new_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker)
        st.rerun()

if st.sidebar.button("🗑️ Reset List"):
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS"]
    st.rerun()

# --- MAIN DASHBOARD ---
st.title("Institutional Sniper Dashboard")

if st.session_state.watchlist:
    selected = st.selectbox("Select Stock to Analyze", st.session_state.watchlist)
    
    if selected:
        try:
            # Fetch Live Data
            with st.spinner(f'Fetching {selected}...'):
                df = yf.download(selected, period="2y", interval="1d", progress=False)
            
            if df.empty or len(df) < 200:
                st.error(f"⚠️ No data found for {selected}. It might be delisted or the ticker is wrong.")
            else:
                # Calculations
                df['EMA200'] = ta.ema(df['Close'], length=200)
                df['EMA50'] = ta.ema(df['Close'], length=50)
                stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
                df = pd.concat([df, stoch], axis=1)
                
                curr = df.iloc[-1]
                prev = df.iloc[-2]

                # --- INTERACTIVE CHART ---
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='orange', width=2), name="EMA 200"))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='blue', width=1), name="EMA 50"))
                
                fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, b=10, t=40))
                st.plotly_chart(fig, use_container_width=True)

                # --- SNIPER ANALYSIS ---
                st.markdown("---")
                is_uptrend = curr['Close'] > curr['EMA200']
                is_oversold = curr['STOCHk_14_3_3'] < 25
                is_trigger = curr['STOCHk_14_3_3'] > curr['STOCHd_14_3_3'] and prev['STOCHk_14_3_3'] <= prev['STOCHd_14_3_3']

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Strategy Check")
                    if is_uptrend and is_oversold and is_trigger:
                        st.success("🎯 A++ SETUP DETECTED!")
                    elif is_uptrend and is_oversold:
                        st.info("🟡 Pullback in Trend: Waiting for Stoch Crossover.")
                    else:
                        st.warning("⚪ Waiting for Conditions.")

                with col2:
                    st.subheader("Execution Plan")
                    entry = curr['Close']
                    sl = curr['Low']
                    risk_per_share = entry - sl
                    qty = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
                    
                    if qty > 0:
                        st.write(f"**Action:** Buy {qty} shares")
                        st.write(f"**Stop Loss:** ₹{round(sl, 2)}")
                        st.write(f"**Target (1:2):** ₹{round(entry + (risk_per_share * 2), 2)}")
                    else:
                        st.write("Calculation error. Price might be at SL level.")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
