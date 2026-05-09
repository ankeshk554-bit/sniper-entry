import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")

try:
    import ta
except:
    ta = None

# ------------------------------------------------------------
# BASIC UNIVERSE (you can extend this)
# ------------------------------------------------------------
NIFTY200 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","HINDUNILVR.NS","ITC.NS","LT.NS",
    "SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","HCLTECH.NS","ASIANPAINT.NS","MARUTI.NS","AXISBANK.NS",
    "SUNPHARMA.NS","BAJFINANCE.NS","ULTRACEMCO.NS","WIPRO.NS"
]

# ------------------------------------------------------------
# SESSION SETTINGS
# ------------------------------------------------------------
if "settings" not in st.session_state:
    st.session_state["settings"] = {
        "default_tf": "1d",
        "default_years": 2,
        "default_risk": 2000,
        "show_rsi": True,
        "show_macd": True,
        "show_ema": True,
        "show_avwap": True,
        "show_divergences": True,
        "show_volume": True
    }

# ------------------------------------------------------------
# DATA LOADER
# ------------------------------------------------------------
@st.cache_data
def load_data(ticker, interval="1d", years=2):
    try:
        df = yf.download(
            ticker,
            period=f"{years}y",
            interval=interval,
            auto_adjust=True,
            progress=False
        )
        return df
    except:
        return pd.DataFrame()

# ------------------------------------------------------------
# INDICATORS (EMA21, EMA200, RSI, MACD)
# ------------------------------------------------------------
def compute_indicators(df):
    df = df.copy()
    if df.empty:
        return df

    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    if ta is not None:
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
        macd = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
        df["MACD_HIST"] = macd.macd_diff()
    else:
        df["RSI"] = 50
        df["MACD"] = 0
        df["MACD_SIGNAL"] = 0
        df["MACD_HIST"] = 0

    return df

# ------------------------------------------------------------
# DIVERGENCES (CLASSIC + HIDDEN)
# ------------------------------------------------------------
def compute_divergences(df):
    df = df.copy()
    if df.empty:
        return [], [], [], []

    bull = []
    bear = []
    hidden_bull = []
    hidden_bear = []

    for i in range(2, len(df)):
        # Classic Bullish
        if df["Low"].iloc[i] < df["Low"].iloc[i-1] and df["RSI"].iloc[i] > df["RSI"].iloc[i-1]:
            bull.append((df.index[i], df["Low"].iloc[i]))

        # Classic Bearish
        if df["High"].iloc[i] > df["High"].iloc[i-1] and df["RSI"].iloc[i] < df["RSI"].iloc[i-1]:
            bear.append((df.index[i], df["High"].iloc[i]))

        # Hidden Bullish
        if df["Low"].iloc[i] > df["Low"].iloc[i-1] and df["RSI"].iloc[i] < df["RSI"].iloc[i-1]:
            hidden_bull.append((df.index[i], df["Low"].iloc[i]))

        # Hidden Bearish
        if df["High"].iloc[i] < df["High"].iloc[i-1] and df["RSI"].iloc[i] > df["RSI"].iloc[i-1]:
            hidden_bear.append((df.index[i], df["High"].iloc[i]))

    return bull, bear, hidden_bull, hidden_bear

# ------------------------------------------------------------
# PULLBACK SETUP (BREAKOUT + HIGH VOL + EMA21 + RSI 50–68)
# ------------------------------------------------------------
def detect_pullback_setups(df):
    df = df.copy()
    setups = []

    if df.empty or "Volume" not in df.columns:
        return setups

    for i in range(21, len(df)):
        window = df.iloc[i-20:i]
        hh20 = window["High"].max()
        avg_vol = window["Volume"].mean()

        close_i = df["Close"].iloc[i]
        high_i = df["High"].iloc[i]
        low_i = df["Low"].iloc[i]
        open_i = df["Open"].iloc[i]
        vol_i = df["Volume"].iloc[i]
        ema21_i = df["EMA21"].iloc[i]
        rsi_i = df["RSI"].iloc[i]

        breakout = close_i > hh20
        high_vol = vol_i > 1.5 * avg_vol
        touch_ema21 = low_i <= ema21_i <= high_i
        rsi_ok = 50 <= rsi_i <= 68
        bullish = close_i > open_i

        if breakout and high_vol and touch_ema21 and rsi_ok and bullish:
            setups.append((df.index[i], close_i))

    return setups

# ------------------------------------------------------------
# AVWAP (10-BAR MAJOR SWING HIGH/LOW)
# ------------------------------------------------------------
def find_major_swings(df, lookback=10):
    swing_high = None
    swing_low = None

    for i in range(lookback, len(df)-lookback):
        if df["High"].iloc[i] == df["High"].iloc[i-lookback:i+lookback+1].max():
            swing_high = df.index[i]
        if df["Low"].iloc[i] == df["Low"].iloc[i-lookback:i+lookback+1].min():
            swing_low = df.index[i]

    return swing_high, swing_low

def compute_avwap(df, anchor_index):
    if anchor_index is None:
        return pd.Series([np.nan] * len(df), index=df.index)

    df2 = df.loc[anchor_index:].copy()
    pv = (df2["Close"] * df2["Volume"]).cumsum()
    v = df2["Volume"].cumsum()
    avwap = pv / v

    full = pd.Series([np.nan] * len(df), index=df.index)
    full.loc[anchor_index:] = avwap
    return full

# ------------------------------------------------------------
# WEEKLY TREND FILTER
# ------------------------------------------------------------
def passes_weekly_trend_filter(ticker):
    df_w = load_data(ticker, interval="1wk", years=1)
    if df_w.empty:
        return False
    df_w = compute_indicators(df_w)
    try:
        return df_w["Close"].iloc[-1] > df_w["EMA200"].iloc[-1] and df_w["RSI"].iloc[-1] > 50
    except:
        return False

# ------------------------------------------------------------
# SCREENER ENGINE
# ------------------------------------------------------------
def run_screener(tickers, interval_s, fresh_only=True, use_trend=True):
    results = []

    for t in tickers:
        df = load_data(t, interval=interval_s, years=st.session_state["settings"]["default_years"])
        if df.empty:
            continue

        df = compute_indicators(df)

        if use_trend and not passes_weekly_trend_filter(t):
            continue

        bull, bear, hidden_bull, hidden_bear = compute_divergences(df)
        pullbacks = detect_pullback_setups(df)

        if fresh_only:
            last_idx = df.index[-3:]
            bull = [(d, p) for (d, p) in bull if d in last_idx]
            bear = [(d, p) for (d, p) in bear if d in last_idx]
            hidden_bull = [(d, p) for (d, p) in hidden_bull if d in last_idx]
            hidden_bear = [(d, p) for (d, p) in hidden_bear if d in last_idx]
            pullbacks = [(d, p) for (d, p) in pullbacks if d in last_idx]

        signal_type = None
        signal_date = None
        strength = 0
        bias = "Neutral"

        if pullbacks:
            signal_type = "PULLBACK"
            signal_date = pullbacks[-1][0]
            strength = 120
            bias = "Bullish"
        elif bull:
            signal_type = "BULLISH"
            signal_date = bull[-1][0]
            strength = 100
            bias = "Bullish"
        elif hidden_bull:
            signal_type = "HIDDEN_BULL"
            signal_date = hidden_bull[-1][0]
            strength = 90
            bias = "Bullish"
        elif bear:
            signal_type = "BEARISH"
            signal_date = bear[-1][0]
            strength = 100
            bias = "Bearish"
        elif hidden_bear:
            signal_type = "HIDDEN_BEAR"
            signal_date = hidden_bear[-1][0]
            strength = 90
            bias = "Bearish"

        if signal_type is not None:
            results.append({
                "Ticker": t,
                "SignalType": signal_type,
                "SignalDate": signal_date,
                "Strength": strength,
                "Bias": bias
            })

    if results:
        return pd.DataFrame(results)
    return pd.DataFrame()

# ------------------------------------------------------------
# CHART ENGINE (PRICE + VOLUME + RSI + MACD)
# ------------------------------------------------------------
def plot_ultra_chart(df, bull_divs, bear_divs, hidden_bull, hidden_bear, pullbacks):
    df = df.copy()
    if df.empty:
        return go.Figure()

    swing_high, swing_low = find_major_swings(df, lookback=10)
    df["AVWAP_HIGH"] = compute_avwap(df, swing_high)
    df["AVWAP_LOW"] = compute_avwap(df, swing_low)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.55, 0.15, 0.15, 0.15]
    )

    # PRICE
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA21"],
        mode="lines", line=dict(color="#00e5ff", width=1.5),
        name="EMA21"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA200"],
        mode="lines", line=dict(color="#ff9100", width=1.5),
        name="EMA200"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["AVWAP_HIGH"],
        mode="lines", line=dict(color="#ff1744", width=2),
        name="AVWAP High"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["AVWAP_LOW"],
        mode="lines", line=dict(color="#00e676", width=2),
        name="AVWAP Low"
    ), row=1, col=1)

    for dt, price in bull_divs:
        if dt in df.index:
            fig.add_trace(go.Scatter(
                x=[dt], y=[price],
                mode="markers",
                marker=dict(color="#4caf50", size=10, symbol="triangle-up"),
                name="Bull Div"
            ), row=1, col=1)

    for dt, price in bear_divs:
        if dt in df.index:
            fig.add_trace(go.Scatter(
                x=[dt], y=[price],
                mode="markers",
                marker=dict(color="#f44336", size=10, symbol="triangle-down"),
                name="Bear Div"
            ), row=1, col=1)

    for dt, price in pullbacks:
        if dt in df.index:
            fig.add_trace(go.Scatter(
                x=[dt], y=[price],
                mode="markers",
                marker=dict(color="#FFD700", size=12, symbol="star"),
                name="Pullback"
            ), row=1, col=1)

    # VOLUME
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        marker_color=np.where(df["Close"] >= df["Open"], "#4caf50", "#f44336"),
        name="Volume"
    ), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        mode="lines", line=dict(color="#D4AF37", width=2),
        name="RSI"
    ), row=3, col=1)

    fig.add_hline(y=70, line=dict(color="red", dash="dot"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="green", dash="dot"), row=3, col=1)
    fig.add_hline(y=50, line=dict(color="white", dash="dot"), row=3, col=1)

    # MACD
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        mode="lines", line=dict(color="#00e5ff", width=2),
        name="MACD"
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_SIGNAL"],
        mode="lines", line=dict(color="#ff9100", width=2),
        name="Signal"
    ), row=4, col=1)

    fig.add_trace(go.Bar(
        x=df.index,
        y=df["MACD_HIST"],
        marker_color=np.where(df["MACD_HIST"] >= 0, "#4caf50", "#f44336"),
        name="Histogram"
    ), row=4, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=1000,
        showlegend=False
    )

    return fig

# ------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="Sniper Divergence Terminal", layout="wide")
    st.title("Sniper Divergence Terminal – AVWAP + RSI + MACD")

    col1, col2, col3 = st.columns(3)
    with col1:
        universe = st.selectbox("Universe", ["NIFTY200", "Custom"])
    with col2:
        interval_s = st.selectbox("Timeframe", ["1d", "1h", "15m", "1wk"], index=0)
    with col3:
        fresh_only = st.checkbox("Fresh signals (last 3 candles)", value=True)

    if universe == "Custom":
        custom = st.text_input("Enter tickers (comma separated)", "INFY.NS, TCS.NS")
        tickers = [x.strip().upper() for x in custom.split(",") if x.strip()]
    else:
        tickers = NIFTY200

    use_trend = st.checkbox("Use Weekly Trend Filter (EMA200 + RSI>50)", value=True)

    if st.button("Run Screener"):
        df_res = run_screener(tickers, interval_s, fresh_only=fresh_only, use_trend=use_trend)
        if df_res.empty:
            st.warning("No setups found.")
        else:
            st.dataframe(df_res, use_container_width=True)
            st.session_state["last_screener"] = df_res
    else:
        df_res = st.session_state.get("last_screener", pd.DataFrame())
        if not df_res.empty:
            st.dataframe(df_res, use_container_width=True)

    st.markdown("---")
    st.subheader("Chart")

    ticker = st.text_input("Ticker for chart", "INFY.NS")
    if st.button("Load Chart"):
        df = load_data(ticker, interval=interval_s, years=st.session_state["settings"]["default_years"])
        if df.empty:
            st.error("No data for this ticker/timeframe.")
        else:
            df = compute_indicators(df)
            bull, bear, hidden_bull, hidden_bear = compute_divergences(df)
            pullbacks = detect_pullback_setups(df)
            fig = plot_ultra_chart(df, bull, bear, hidden_bull, hidden_bear, pullbacks)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
