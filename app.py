import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. TRADINGVIEW PRO INTERFACE ---
st.set_page_config(page_title="Sniper Terminal v6.0", layout="wide")

st.markdown("""
    <style>
    .main { background: #131722; } /* TradingView Original Dark */
    [data-testid="stSidebar"] { background-color: #171b26; border-right: 1px solid #363c4e; }
    
    /* Sniper HUD */
    .stMetric { background-color: #1e222d; border: 1px solid #363c4e; padding: 15px; border-radius: 4px; }
    div[data-testid="stMetricValue"] { color: #2962ff !important; font-size: 28px !important; }
    
    /* Top Bar Status */
    .tv-status {
        background: #1e222d;
        color: #d1d4dc;
        padding: 10px;
        border-radius: 4px;
        border-left: 5px solid #2962ff;
        margin-bottom: 20px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIC ENGINE ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "TITAN.NS"]

st.sidebar.title("📡 SNIPER v6.0")
risk_amt = st.sidebar.number_input("Risk Capital (₹)", value=2000)

# Ticker Selection
ticker = st.sidebar.selectbox("Active Chart", st.session_state.watchlist)
tf_options = {"Daily": "1d", "Weekly": "1wk", "1 Hour": "1h", "15 Min": "15m"}
interval = tf_options[st.sidebar.selectbox("Timeframe", list(tf_options.keys()))]

st.sidebar.markdown("---")
new_tick = st.sidebar.text_input("Add Ticker (e.g. SBIN.NS)").upper()
if st.sidebar.button("➕ Update Watchlist"):
    if new_tick and new_tick not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_tick)
        st.rerun()

# --- 3. DATA PROCESSING (THE SCALE FIX) ---
def get_clean_data(symbol, interval):
    period = "max" if interval in ["1d", "1wk"] else "60d"
    # Group_by='ticker' is the key to stopping the MultiIndex bug
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty: return None
    
    # Flatten columns to ensure 'Close' is just 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Ensure all required columns are present
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # Technicals
    df['EMA200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # AVWAP Calculation (Pure Price logic)
    high_idx = df['High'].idxmax()
    low_idx = df['Low'].idxmin()
    def avwap(anchor):
        tmp = df.loc[anchor:].copy()
        return (tmp['Close'] * tmp['Volume']).cumsum() / tmp['Volume'].cumsum()
    
    df['AVWAP_H'] = avwap(high_idx)
    df['AVWAP_L'] = avwap(low_idx)
    
    return df

# --- 4. THE TERMINAL DISPLAY ---
if ticker:
    df = get_clean_data(ticker, interval)
    if df is not None:
        curr = df.iloc[-1]
        
        # --- TOP HUD ---
        st.markdown(f'<div class="tv-status">{ticker} • {interval} • Institutional Sniper Mode Activated</div>', unsafe_allow_html=True)
        
        # Math for your ₹2000 Risk
        sl_price = curr['Low']
        risk_per_share = curr['Close'] - sl_price
        qty = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Live Price", f"₹{round(curr['Close'], 2)}")
        c2.metric("Buy Qty", f"{qty} Units")
        c3.metric("Stop Loss", f"₹{round(sl_price, 2)}")
        c4.metric("Target (1:2)", f"₹{round(curr['Close'] + (risk_per_share * 2), 2)}")

        # --- 5. WORLD CLASS CHARTING ---
        # 3 Rows: Price (70%), Volume (15%), RSI (15%)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.02, 
                            row_width=[0.15, 0.15, 0.7])

        # A. MAIN CHART (Right-side Axis)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                                     name="Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
                                     increasing_fillcolor='#26a69a', decreasing_fillcolor='#ef5350'), row=1, col=1)
        
        # Pro Overlays
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='#2962ff', width=1.5, dash='dot'), name="EMA 200"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_H'], line=dict(color='#ef5350', width=2), name="Resistance"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP_L'], line=dict(color='#26a69a', width=2), name="Support"), row=1, col=1)

        # B. VOLUME (TradingView Style)
        colors = ['#26a69a' if df['Open'].iloc[i] < df['Close'].iloc[i] else '#ef5350' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="Volume", opacity=0.5), row=2, col=1)

        # C. RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#787b86', width=2), name="RSI"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", opacity=0.3, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", opacity=0.3, row=3, col=1)

        # FINAL TV-STYLING
        fig.update_layout(
            template="plotly_dark",
            height=850,
            xaxis_rangeslider_visible=False,
            showlegend=False,
            paper_bgcolor="#131722",
            plot_bgcolor="#131722",
            margin=dict(l=0, r=50, b=0, t=10), # Right margin for labels
            hovermode='x unified'
        )
        
        # Grid Styling
        fig.update_xaxes(showgrid=True, gridcolor="#1f222d", zeroline=False, range=[df.index[-120], df.index[-1]])
        fig.update_yaxes(showgrid=True, gridcolor="#1f222d", zeroline=False, side="right")

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.error("Signal Lost: Ticker data is unavailable or the symbol is incorrect.")
