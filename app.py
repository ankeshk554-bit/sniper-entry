import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Ankesh Sniper Pro", layout="wide")

# --- INITIALIZE WATCHLIST ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "BHARTIARTL.NS"]

# --- SIDEBAR: WATCHLIST MANAGER ---
st.sidebar.title("🏹 Sniper Settings")
risk_amt = st.sidebar.number_input("Risk per Trade (₹)", value=2000)

st.sidebar.markdown("---")
st.sidebar.subheader("Manage Watchlist")
new_ticker = st.sidebar.text_input("Add Ticker (e.g. SBIN.NS)").upper()
if st.sidebar.button("➕ Add to List"):
    if new_ticker and new_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker)
        st.rerun()

if st.sidebar.button("🗑️ Clear Watchlist"):
    st.session_state.watchlist = []
    st.rerun()

# --- MAIN INTERFACE: SELECT & ANALYZE ---
st.title("Institutional Sniper Hub")

if not st.session_state.watchlist:
    st.info("Your watchlist is empty. Add some tickers in the sidebar to begin.")
else:
    # Clickable Selection
    selected_stock = st.selectbox("Select Stock to Analyze", st.session_state.watchlist)

    if selected_stock:
        try:
            # Fetch Live Data
            df = yf.download(selected_stock, period="1y", interval="1d", progress=False)
            
            if df.empty:
                st.error("No data found. Ensure the ticker suffix is correct (e.g., .NS)")
            else:
                # Technicals
                df['EMA200'] = ta.ema(df['Close'], length=200)
                df['EMA50'] = ta.ema(df['Close'], length=50)
                stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
                df = pd.concat([df, stoch], axis=1)
                
                curr, prev = df.iloc[-1], df.iloc[-2]

                # --- INTERACTIVE CHART (Plotly) ---
                fig = go.Figure()
                # Candlesticks
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
                # Moving Averages
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='orange', width=2), name="EMA 200"))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='blue', width=1), name="EMA 50"))
                
                fig.update_layout(title=f"{selected_stock} Technical Chart", yaxis_title="Price", height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                # --- SNIPER STRATEGY LOGIC ---
                is_uptrend = curr['Close'] > curr['EMA200']
                is_oversold = curr['STOCHk_14_3_3'] < 25
                is_trigger = curr['STOCHk_14_3_3'] > curr['STOCHd_14_3_3'] and prev['STOCHk_14_3_3'] <= prev['STOCHd_14_3_3']

                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Price", round(curr['Close'], 2))
                    if is_uptrend and is_oversold and is_trigger:
                        st.success("🎯 A++ SETUP DETECTED")
                    else:
                        st.warning("⏳ Waiting for Setup")

                with col2:
                    # Risk Calculation
                    entry = curr['Close']
                    sl = curr['Low']
                    risk_per_share = entry - sl
                    qty = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
                    
                    st.write("### Execution Plan")
                    st.write(f"**Action:** Buy {qty} shares")
                    st.write(f"**Stop Loss:** {round(sl, 2)}")
                    st.write(f"**Target (1:2):** {round(entry + (risk_per_share * 2), 2)}")

        except Exception as e:
            st.error(f"Error fetching {selected_stock}: {e}")
