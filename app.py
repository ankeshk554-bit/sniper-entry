import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE SETUP ---
st.set_page_config(page_title="Institutional Sniper Terminal", layout="wide")

# --- INITIALIZE WATCHLIST ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "TITAN.NS", "NIFTY_50"]

# --- SIDEBAR ---
st.sidebar.title("🏹 Sniper Terminal")
risk_amt = st.sidebar.number_input("Risk per Trade (₹)", value=2000)
tf_options = {"1W": "1wk", "1D": "1d", "4H": "4h", "1H": "1h", "15M": "15m", "5M": "5m"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_options.keys()), index=1)
interval = tf_options[selected_tf]

# --- DIVERGENCE ENGINE ---
def detect_rsi_divergence(df, window=5):
    df['Bullish_Div'] = False
    df['Bearish_Div'] = False
    
    for i in range(window, len(df)):
        # Bullish Divergence: Price Lower Low, RSI Higher Low
        if df['Low'].iloc[i] < df['Low'].iloc[i-window] and df['RSI'].iloc[i] > df['RSI'].iloc[i-window]:
            if df['RSI'].iloc[i] < 40: # Oversold area
                df.at[df.index[i], 'Bullish_Div'] = True
        
        # Bearish Divergence: Price Higher High, RSI Lower High
        if df['High'].iloc[i] > df['High'].iloc[i-window] and df['RSI'].iloc[i] < df['RSI'].iloc[i-window]:
            if df['RSI'].iloc[i] > 60: # Overbought area
                df.at[df.index[i], 'Bearish_Div'] = True
    return df

# --- MAIN INTERFACE ---
ticker = st.sidebar.text_input("🔍 Search Ticker", "RELIANCE.NS").upper()

if ticker:
    try:
        data_period = "2y" if interval in ["1d", "1wk"] else "60d"
        df_raw = yf.download(ticker, period=data_period, interval=interval, progress=False)
        
        if not df_raw.empty:
            df = df_raw.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 1. Calculations
            df['EMA200'] = ta.ema(df['Close'], length=200)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df = detect_rsi_divergence(df)
            
            # AVWAP Calculations
            top_idx, bot_idx = df['High'].argmax(), df['Low'].argmin()
            def calc_avwap(df, idx):
                temp = df.iloc[idx:].copy()
                return (temp['Close'] * temp['Volume']).cumsum() / temp['Volume'].cumsum()
            df['AVWAP_TOP'] = calc_avwap(df, top_idx)
            df['AVWAP_BOT'] = calc_avwap(df, bot_idx)

            # --- PLOTLY SUBPLOTS ---
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, subplot_titles=(f'{ticker} Price', 'Volume', 'RSI'), 
                                row_width=[0.2, 0.2, 0.6])

            # Row 1: Candlesticks + AVWAPs
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_TOP'], line=dict(color='red', width=2), name="Top Supply"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_BOT'], line=dict(color='green', width=2), name="Bot Support"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='white', dash='dot'), name="EMA 200"), row=1, col=1)

            # Row 2: Volume
            colors = ['green' if df['Open'].iloc[i] < df['Close'].iloc[i] else 'red' for i in range(len(df))]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)

            # Row 3: RSI + Divergence
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name="RSI"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            # Divergence Markers
            bull_divs = df[df['Bullish_Div']]
            bear_divs = df[df['Bearish_Div']]
            fig.add_trace(go.Scatter(x=bull_divs.index, y=bull_divs['RSI'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='lime'), name="Bullish Div"), row=3, col=1)
            fig.add_trace(go.Scatter(x=bear_divs.index, y=bear_divs['RSI'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='orange'), name="Bearish Div"), row=3, col=1)

            fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # --- EXECUTION MATH ---
            curr = df.iloc[-1]
            entry, sl = curr['Close'], curr['Low']
            qty = int(risk_amt / (entry - sl)) if entry > sl else 0
            
            st.sidebar.markdown("---")
            st.sidebar.subheader("Trade Plan")
            st.sidebar.write(f"**Action:** Buy {qty} shares")
            st.sidebar.write(f"**Stop Loss:** ₹{round(sl, 2)}")
            st.sidebar.write(f"**Target:** ₹{round(entry + (entry-sl)*2, 2)}")

    except Exception as e:
        st.error(f"Error: {e}")
