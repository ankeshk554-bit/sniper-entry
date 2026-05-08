import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE SETUP ---
st.set_page_config(page_title="Sniper Terminal v2.5", layout="wide")

# --- IPAD UI OPTIMIZATION ---
st.markdown("""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child { width: 320px; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE: WATCHLIST ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "TITAN.NS"]

# --- SIDEBAR: CONTROL PANEL ---
st.sidebar.title("🏹 Control Panel")
risk_amt = st.sidebar.number_input("Risk per Trade (₹)", value=2000, step=100)

st.sidebar.markdown("---")
st.sidebar.subheader("Watchlist Manager")

# Add Ticker Logic
new_ticker = st.sidebar.text_input("Add NSE Stock (e.g. INFIBEAM.NS)").upper()
if st.sidebar.button("➕ Add to Watchlist"):
    if new_ticker and new_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker)
        st.rerun()

# Select Active Ticker
active_ticker = st.sidebar.selectbox("Select Active Ticker", st.session_state.watchlist)

if st.sidebar.button("🗑️ Clear Watchlist"):
    st.session_state.watchlist = ["RELIANCE.NS"]
    st.rerun()

st.sidebar.markdown("---")
tf_options = {"1 Day": "1d", "1 Week": "1wk", "4 Hour": "4h", "1 Hour": "1h"}
selected_tf = st.sidebar.selectbox("Timeframe", list(tf_options.keys()), index=0)
interval = tf_options[selected_tf]

# --- DIVERGENCE ENGINE ---
def calculate_divergence_lines(df, window=5):
    df['Price_Low'] = df['Low'].rolling(window=window*2+1, center=True).min()
    df['Price_High'] = df['High'].rolling(window=window*2+1, center=True).max()
    df['RSI_Low'] = df['RSI'].rolling(window=window*2+1, center=True).min()
    df['RSI_High'] = df['RSI'].rolling(window=window*2+1, center=True).max()
    
    bull_lines, bear_lines = [], []
    lp_low, lp_high = -1, -1

    for i in range(window*2, len(df)):
        if df['Low'].iloc[i] == df['Price_Low'].iloc[i]:
            if lp_low != -1:
                if df['Low'].iloc[i] < df['Low'].iloc[lp_low] and df['RSI'].iloc[i] > df['RSI'].iloc[lp_low]:
                    if df['RSI'].iloc[i] < 40:
                        bull_lines.append(((df.index[lp_low], df['RSI'].iloc[lp_low]), (df.index[i], df['RSI'].iloc[i])))
                        bull_lines.append(((df.index[lp_low], df['Low'].iloc[lp_low]), (df.index[i], df['Low'].iloc[i]), 'price'))
            lp_low = i
        if df['High'].iloc[i] == df['Price_High'].iloc[i]:
            if lp_high != -1:
                if df['High'].iloc[i] > df['High'].iloc[lp_high] and df['RSI'].iloc[i] < df['RSI'].iloc[lp_high]:
                    if df['RSI'].iloc[i] > 60:
                        bear_lines.append(((df.index[lp_high], df['RSI'].iloc[lp_high]), (df.index[i], df['RSI'].iloc[i])))
                        bear_lines.append(((df.index[lp_high], df['High'].iloc[lp_high]), (df.index[i], df['High'].iloc[i]), 'price'))
            lp_high = i
    return bull_lines, bear_lines

# --- MAIN DASHBOARD ---
st.title(f"Institutional Sniper Terminal - {active_ticker}")

if active_ticker:
    try:
        # Buffer period to ensure EMA 200 calculates
        period = "max" if interval in ["1d", "1wk"] else "60d"
        with st.spinner(f'Analyzing {active_ticker}...'):
            raw_df = yf.download(active_ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            
            if not raw_df.empty:
                df = raw_df.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Tech Calculations
                df['EMA200'] = ta.ema(df['Close'], length=200)
                df['RSI'] = ta.rsi(df['Close'], length=14)
                bull_lines, bear_lines = calculate_divergence_lines(df)
                
                # AVWAP
                top_idx, bot_idx = df['High'].argmax(), df['Low'].argmin()
                def calc_avwap(df, idx):
                    temp = df.iloc[idx:].copy()
                    return (temp['Close'] * temp['Volume']).cumsum() / temp['Volume'].cumsum()
                df['AVWAP_TOP'] = calc_avwap(df, top_idx)
                df['AVWAP_BOT'] = calc_avwap(df, bot_idx)

                # --- PLOTLY SUBPLOTS ---
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.2, 0.6],
                                    subplot_titles=('Price Structure + AVWAP', 'Volume Profile', 'RSI Divergence Lines'))

                # Price Row
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_TOP'], line=dict(color='#ff4b4b', width=2), name="Supply"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_BOT'], line=dict(color='#00ff41', width=2), name="Support"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='white', dash='dot', width=1), name="EMA 200"), row=1, col=1)
                
                # Divergence Price Lines
                for l in bull_lines:
                    if len(l) == 3: fig.add_trace(go.Scatter(x=[l[0][0], l[1][0]], y=[l[0][1], l[1][1]], mode='lines+markers', line=dict(color='lime', width=1), showlegend=False), row=1, col=1)
                for l in bear_lines:
                    if len(l) == 3: fig.add_trace(go.Scatter(x=[l[0][0], l[1][0]], y=[l[0][1], l[1][1]], mode='lines+markers', line=dict(color='red', width=1), showlegend=False), row=1, col=1)

                # Volume Row
                v_colors = ['#1a9c37' if df['Open'].iloc[i] < df['Close'].iloc[i] else '#d92c2c' for i in range(len(df))]
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=v_colors, name="Volume"), row=2, col=1)

                # RSI Row
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#ab63fa', width=2), name="RSI"), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                for l in bull_lines:
                    if len(l) == 2: fig.add_trace(go.Scatter(x=[l[0][0], l[1][0]], y=[l[0][1], l[1][1]], mode='lines+markers', line=dict(color='lime', width=3), showlegend=False), row=3, col=1)
                for l in bear_lines:
                    if len(l) == 2: fig.add_trace(go.Scatter(x=[l[0][0], l[1][0]], y=[l[0][1], l[1][1]], mode='lines+markers', line=dict(color='red', width=3), showlegend=False), row=3, col=1)

                fig.update_layout(template="plotly_dark", height=850, xaxis_rangeslider_visible=False, showlegend=False, margin=dict(l=10, r=10, b=10, t=50))
                # Zoom to last 150 bars for better iPad viewing
                fig.update_xaxes(range=[df.index[-150], df.index[-1]])
                st.plotly_chart(fig, use_container_width=True)

                # --- RISK REPORT ---
                st.markdown("---")
                curr = df.iloc[-1]
                entry, sl = curr['Close'], curr['Low']
                qty = int(risk_amt / (entry - sl)) if entry > sl else 0
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Price", f"₹{round(entry, 2)}")
                c2.metric("Buy Quantity", f"{qty} Shares")
                c3.metric("Stop Loss", f"₹{round(sl, 2)}")
                c4.metric("Target (1:2)", f"₹{round(entry + (entry-sl)*2), 2}")

    except Exception as e:
        st.error(f"Waiting for valid data... {e}")
