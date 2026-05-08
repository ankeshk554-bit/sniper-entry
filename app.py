import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd

st.set_page_config(page_title="Institutional Sniper", layout="wide")

st.title("🏹 Institutional Sniper Dashboard")

# Risk Input
risk_amt = st.sidebar.number_input("Risk Amount (₹)", value=2000)
watchlist = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "TITAN.NS", "BHARTIARTL.NS", "TATASTEEL.NS", "KOTAKBANK.NS", "ICICIBANK.NS", "INFY.NS"]

if st.button("🚀 Run Scan"):
    results = []
    for s in watchlist:
        try:
            # Fetch data with a small buffer for safety
            df = yf.download(s, period="1y", interval="1d", progress=False)
            if df.empty or len(df) < 150:
                continue
            
            # Indicators
            df['EMA200'] = ta.ema(df['Close'], length=200)
            df['EMA50'] = ta.ema(df['Close'], length=50)
            stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
            df = pd.concat([df, stoch], axis=1)
            
            curr, prev = df.iloc[-1], df.iloc[-2]
            
            # A++ Strategy Logic
            is_uptrend = curr['Close'] > curr['EMA200']
            is_oversold = curr['STOCHk_14_3_3'] < 25
            is_trigger = curr['STOCHk_14_3_3'] > curr['STOCHd_14_3_3'] and prev['STOCHk_14_3_3'] <= prev['STOCHd_14_3_3']
            
            if is_uptrend and is_oversold and is_trigger:
                entry = curr['Close']
                sl = curr['Low']
                # Risk Formula: Q = Risk / (Entry - SL)
                risk_per_share = entry - sl
                qty = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
                
                results.append({
                    "Stock": s.replace(".NS", ""),
                    "Entry": round(entry, 2),
                    "Stop Loss": round(sl, 2),
                    "Quantity": qty,
                    "Target (1:2)": round(entry + (risk_per_share * 2), 2)
                })
        except Exception:
            continue
            
    if results:
        st.success("A++ Signals Found!")
        st.table(pd.DataFrame(results))
    else:
        st.info("No A++ setups found. Patience protects your capital.")
