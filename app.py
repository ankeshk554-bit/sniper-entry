import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd

st.set_page_config(page_title="Institutional Sniper Search", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #008000; color: white; }
    .stTextInput>div>div>input { background-color: #161b22; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏹 A++ Institutional Sniper Search")

# --- SIDEBAR: RISK SETTINGS ---
st.sidebar.header("Risk Settings")
risk_amt = st.sidebar.number_input("Risk Amount (₹/$)", value=2000)
st.sidebar.markdown("---")
st.sidebar.write("Note: For Indian stocks, add **.NS** (e.g., RELIANCE.NS)")

# --- MAIN INTERFACE: SEARCH ---
search_ticker = st.text_input("Enter Ticker Symbol to Search", value="RELIANCE.NS").upper()

if st.button(f"Analyze {search_ticker}"):
    try:
        # Pulling live-ready data (1-day interval for the latest close)
        df = yf.download(search_ticker, period="1y", interval="1d", progress=False)
        
        if df.empty or len(df) < 150:
            st.error(f"Could not find sufficient data for {search_ticker}. Check the symbol.")
        else:
            # Calculate Indicators
            df['EMA200'] = ta.ema(df['Close'], length=200)
            df['EMA50'] = ta.ema(df['Close'], length=50)
            stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
            df = pd.concat([df, stoch], axis=1)
            
            curr, prev = df.iloc[-1], df.iloc[-2]
            
            # Display Current Stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", round(curr['Close'], 2))
            col2.metric("200 EMA", round(curr['EMA200'], 2))
            col3.metric("Stoch %K", round(curr['STOCHk_14_3_3'], 2))

            # A++ Confluence Check
            is_uptrend = curr['Close'] > curr['EMA200']
            is_oversold = curr['STOCHk_14_3_3'] < 25
            is_trigger = curr['STOCHk_14_3_3'] > curr['STOCHd_14_3_3'] and prev['STOCHk_14_3_3'] <= prev['STOCHd_14_3_3']
            
            if is_uptrend and is_oversold and is_trigger:
                st.success("✅ A++ Sniper Setup Confirmed!")
                
                # Math: Risk per share = Entry - Stop Loss
                entry = curr['Close']
                sl = curr['Low']
                risk_per_share = entry - sl
                qty = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
                
                # Trade Plan
                st.write("### 📝 Trade Plan")
                plan_df = pd.DataFrame({
                    "Parameter": ["Quantity", "Entry", "Stop Loss", "Target (1:2)"],
                    "Value": [qty, round(entry, 2), round(sl, 2), round(entry + (risk_per_share * 2), 2)]
                })
                st.table(plan_df)
            else:
                st.info("No A++ setup found. Criteria not met (Needs: Above 200 EMA, Stoch < 25, and Crossover).")
                
            # Show Recent Data Chart
            st.line_chart(df[['Close', 'EMA200']])

    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- AUTOMATIC WATCHLIST SECTION ---
st.markdown("---")
st.subheader("Quick Watchlist Check")
if st.button("Run Watchlist Scan"):
    # (Existing watchlist code goes here)
    pass
