import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE SETUP ---
st.set_page_config(page_title="Ankesh Sniper Terminal v2.0", layout="wide")

# --- CUSTOM CSS FOR IPAD OPTIMIZATION ---
st.markdown("""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child { width: 300px; }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child { width: 300px; margin-left: -300px; }
    .stMetric { background-color: #161b22; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "TITAN.NS"]

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.title("🏹 Control Panel")
risk_amt = st.sidebar.number_input("Risk per Trade (₹)", value=2000, step=100)
tf_options = {"1 Day": "1d", "1 Week": "1wk", "4 Hour": "4h", "1 Hour": "1h", "15 Min": "15m"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_options.keys()), index=0)
interval = tf_options[selected_tf]

st.sidebar.markdown("---")
st.sidebar.subheader("Ticker Search")
ticker = st.sidebar.text_input("Enter Ticker (e.g. SBIN.NS)", "RELIANCE.NS").upper()

# --- COMPLEX DIVERGENCE LINE CALCULATOR ---
def calculate_divergence_lines(df, window=5):
    # Find local price & RSI peaks/troughs
    df['Price_Low'] = df['Low'].rolling(window=window*2+1, center=True).min()
    df['Price_High'] = df['High'].rolling(window=window*2+1, center=True).max()
    df['RSI_Low'] = df['RSI'].rolling(window=window*2+1, center=True).min()
    df['RSI_High'] = df['RSI'].rolling(window=window*2+1, center=True).max()
    
    bull_lines, bear_lines = [], []
    last_p_low, last_p_high = -1, -1

    for i in range(window*2, len(df)):
        # Calculate Trough coordinates (Bullish)
        if df['Low'].iloc[i] == df['Price_Low'].iloc[i]:
            if last_p_low != -1:
                # Math: Price LL, RSI HL
                if df['Low'].iloc[i] < df['Low'].iloc[last_p_low] and df['RSI'].iloc[i] > df['RSI'].iloc[last_p_low]:
                    if df['RSI'].iloc[i] < 35: # Quality oversold check
                        bull_lines.append(((df.index[last_p_low], df['RSI'].iloc[last_p_low]), (df.index[i], df['RSI'].iloc[i])))
                        bull_lines.append(((df.index[last_p_low], df['Low'].iloc[last_p_low]), (df.index[i], df['Low'].iloc[i]), 'price')) # Price overlay
            last_p_low = i
            
        # Calculate Peak coordinates (Bearish)
        if df['High'].iloc[i] == df['Price_High'].iloc[i]:
            if last_p_high != -1:
                # Math: Price HH, RSI LH
                if df['High'].iloc[i] > df['High'].iloc[last_p_high] and df['RSI'].iloc[i] < df['RSI'].iloc[last_p_high]:
                    if df['RSI'].iloc[i] > 65: # Quality overbought check
                        bear_lines.append(((df.index[last_p_high], df['RSI'].iloc[last_p_high]), (df.index[i], df['RSI'].iloc[i])))
                        bear_lines.append(((df.index[last_p_high], df['High'].iloc[last_p_high]), (df.index[i], df['High'].iloc[i]), 'price')) # Price overlay
            last_p_high = i
            
    return bull_lines, bear_lines

# --- MAIN DASHBOARD EXECUTION ---
st.title(f"Institutional Sniper Terminal - {ticker}")

if ticker:
    try:
        data_period = "2y" if interval in ["1d", "1wk"] else "60d"
        with st.spinner(f'Analyzing {ticker} Structure...'):
            raw_df = yf.download(ticker, period=data_period, interval=interval, progress=False, auto_adjust=True)
            
            if not raw_df.empty:
                df = raw_df.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Calculations
                df['EMA200'] = ta.ema(df['Close'], length=200)
                df['RSI'] = ta.rsi(df['Close'], length=14)
                bull_lines, bear_lines = calculate_divergence_lines(df)
                
                # Anchored VWAP (Topsupply & Bot Support)
                top_idx, bot_idx = df['High'].argmax(), df['Low'].argmin()
                def calc_avwap(df, idx):
                    temp = df.iloc[idx:].copy()
                    return (temp['Close'] * temp['Volume']).cumsum() / temp['Volume'].cumsum()
                df['AVWAP_TOP'] = calc_avwap(df, top_idx)
                df['AVWAP_BOT'] = calc_avwap(df, bot_idx)

                # --- PLOTLY TERMINAL SUBPLOTS ---
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.03, row_width=[0.2, 0.2, 0.6],
                                    subplot_titles=(f'Institutional Price Structure ({selected_tf})', 'Volume Profile', 'RSI with Fixed Line Divergence'))

                # Row 1: Price + Overlays
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_TOP'], line=dict(color='red', width=2), name="Top Supply"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_BOT'], line=dict(color='green', width=2), name="Bot Support"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='white', dash='dot', width=1), name="EMA 200"), row=1, col=1)
                
                # Divergence Overlays on Price (Connecting Troughs and Peaks)
                for line in bull_lines:
                    if len(line) == 3 and line[2] == 'price':
                        fig.add_trace(go.Scatter(x=[line[0][0], line[1][0]], y=[line[0][1], line[1][1]], mode='lines+markers', line=dict(color='lime', width=1), showlegend=False), row=1, col=1)
                for line in bear_lines:
                    if len(line) == 3 and line[2] == 'price':
                        fig.add_trace(go.Scatter(x=[line[0][0], line[1][0]], y=[line[0][1], line[1][1]], mode='lines+markers', line=dict(color='red', width=1), showlegend=False), row=1, col=1)

                # Row 2: Volume
                colors = ['#1a9c37' if df['Open'].iloc[i] < df['Close'].iloc[i] else '#d92c2c' for i in range(len(df))]
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)

                # Row 3: RSI + Divergence Lines
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name="RSI"), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", row=3, col=1)
                
                # DRAW THE DIVERGENCE LINES (The core fix)
                for line in bull_lines:
                    if len(line) == 2: # Plot standard line on RSI
                        fig.add_trace(go.Scatter(x=[line[0][0], line[1][0]], y=[line[0][1], line[1][1]], mode='lines+markers', line=dict(color='lime', width=3), name="Bull Div Line"), row=3, col=1)
                for line in bear_lines:
                    if len(line) == 2: # Plot standard line on RSI
                        fig.add_trace(go.Scatter(x=[line[0][0], line[1][0]], y=[line[0][1], line[1][1]], mode='lines+markers', line=dict(color='red', width=3), name="Bear Div Line"), row=3, col=1)

                # Dashboard Layout Optimization
                fig.update_layout(template="plotly_dark", height=850, xaxis_rangeslider_visible=False, showlegend=False, 
                                  margin=dict(l=10, r=10, b=10, t=50))
                st.plotly_chart(fig, use_container_width=True)

                # --- INSTANT EXECUTION REPORT (Side Metric) ---
                curr = df.iloc[-1]
                entry, sl = curr['Close'], curr['Low']
                diff = entry - sl
                qty = int(risk_amt / diff) if diff > 0 else 0
                
                st.markdown("---")
                st.subheader("Monday Morning Execution Plan")
                colA, colB, colC, colD = st.columns(4)
                colA.metric(f"Current {ticker} Price", f"₹{round(entry, 2)}")
                colB.metric("Qty (Risk ₹2k)", f"{qty} Shares")
                colC.metric("Stop Loss Level", f"₹{round(sl, 2)}")
                colD.metric("Target Level (1:2)", f"₹{round(entry + (diff * 2), 2)}")

    except Exception as e:
        st.error(f"Analysis Error: {e}")
