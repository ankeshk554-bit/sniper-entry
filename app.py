import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd

# Page Setup for iPad/Mobile
st.set_page_config(page_title="Ankesh Sniper Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    stTable { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏹 Ankesh's Institutional Sniper")
st.subheader("High-Conviction A++ Screener")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Risk Management")
risk_per_trade = st.sidebar.number_input("Risk per Trade (₹)", value=2000)
st.sidebar.markdown("---")
st.sidebar.info("This tool filters for Institutional Trend + Retail Panic setups.")

# --- THE LIST (Nifty Heavyweights) ---
watchlist = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "TITAN.NS", "BHARTIARTL.NS", "TATASTEEL.NS", 
             "KOTAKBANK.NS", "ICICIBANK.NS", "INFY.NS", "SUNPHARMA.NS", "LT.NS", "ITC.NS"]

if st.button("🚀 Run Market Scan"):
    results = []
    progress_bar = st.progress(0)
    
    for i, s in enumerate(watchlist):
        try:
            # Pulling Data
            df = yf.download(s, period="1y", interval="1d", progress=False)
            if len(df) < 150: continue
            
            # Technical Indicators
            df['EMA200'] = ta.ema(df['Close'], length=200)
            df['EMA50'] = ta.ema(df['Close'], length=50)
            stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
            df = pd.concat([df, stoch], axis=1)
            
            curr, prev = df.iloc[-1], df.iloc[-2]
            
            # A++ Confluence Logic
            is_uptrend = curr['Close'] > curr['EMA200'] and curr['Close'] > curr['EMA50']
            is_oversold = curr['STOCHk_14_3_3'] < 25
            is_trigger = curr['STOCHk_14_3_3'] > curr['STOCHd_14_3_3'] and prev['STOCHk_14_3_3'] <= prev['STOCHd_14_3_3']
            
            if is_uptrend and is_oversold and is_trigger:
                entry = curr['Close']
                sl = curr['Low']
                # Risk Mgmt Math
                risk_per_share = entry - sl
                qty = int(risk_per_trade / risk_per_share) if risk_per_share > 0 else 0
                target = entry + (risk_per_share * 2)
                
                results.append({
                    "Stock": s.replace(".NS", ""),
                    "Price": round(entry, 2),
                    "Stop Loss": round(sl, 2),
                    "Qty to Buy": qty,
                    "Target (1:2)": round(target, 2),
                    "Total Value": round(qty * entry, 2)
                })
        except: continue
        progress_bar.progress((i + 1) / len(watchlist))

    if results:
        st.success(f"Found {len(results)} A++ Setups!")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.warning("No A++ setups found. Keep your capital safe in the bank today.")

# --- FOOTER ---
st.markdown("---")
st.caption("Strategy: Institutional Trend (EMA 200) + Retail Exhaustion (Stoch).")
