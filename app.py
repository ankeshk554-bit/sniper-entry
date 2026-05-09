import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# Optional libs
try:
    import ta
except:
    ta = None

# ============================================================
# UNIVERSE (NIFTY200 PLACEHOLDER)
# ============================================================
# Replace this with your full NIFTY200 list if you want
NIFTY200 = [
    "INFY.NS", "TCS.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS", "LT.NS", "ITC.NS"
]

# ============================================================
# GLOBAL SETTINGS
# ============================================================
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

# ============================================================
# DATA LOADER
# ============================================================
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
    except Exception:
        return pd.DataFrame()

# ============================================================
# INDICATORS (EMA21, EMA200, RSI, MACD)
# ============================================================
def compute_indicators(df):
    df = df.copy()
    if df.empty:
        return df

    # EMA
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # RSI
    if ta is not None:
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    else:
        df["RSI"] = 50

    # MACD (12, 26, 9)
    if ta is not None:
        macd = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
        df["MACD_HIST"] = macd.macd_diff()
    else:
        df["MACD"] = df["MACD_SIGNAL"] = df["MACD_HIST"] = 0

    return df

# ============================================================
# DIVERGENCE ENGINE (CLASSIC + HIDDEN)
# ============================================================
def compute_divergences(df):
    df = sanitize_df(df)

    if df.empty or "RSI" not in df.columns:
        return [], [], [], []

    bull = []
    bear = []
    hidden_bull = []
    hidden_bear = []

    for i in range(2, len(df)):
        # Extract values safely
        try:
            L0 = float(df["Low"].iloc[i])
            L1 = float(df["Low"].iloc[i-1])
            H0 = float(df["High"].iloc[i])
            H1 = float(df["High"].iloc[i-1])
            R0 = float(df["RSI"].iloc[i])
            R1 = float(df["RSI"].iloc[i-1])
        except Exception:
            continue  # skip corrupted rows

        # Skip NaN or invalid values
        if any([
            np.isnan(L0), np.isnan(L1),
            np.isnan(H0), np.isnan(H1),
            np.isnan(R0), np.isnan(R1)
        ]):
            continue

        # Classic Bullish
        if L0 < L1 and R0 > R1:
            bull.append((df.index[i], L0))

        # Classic Bearish
        if H0 > H1 and R0 < R1:
            bear.append((df.index[i], H0))

        # Hidden Bullish
        if L0 > L1 and R0 < R1:
            hidden_bull.append((df.index[i], L0))

        # Hidden Bearish
        if H0 < H1 and R0 > R1:
            hidden_bear.append((df.index[i], H0))

    return bull, bear, hidden_bull, hidden_bear




# ============================================================
# PULLBACK SETUP (HIGH VOLUME + EMA21 + RSI 50–68)
# ============================================================
def detect_pullback_setups(df):
    df = df.copy()
    setups = []

    if df.empty or "Volume" not in df.columns or "EMA21" not in df.columns or "RSI" not in df.columns:
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

# ============================================================
# AVWAP (10-BAR MAJOR SWING HIGH/LOW)
# ============================================================
def find_major_swings(df, lookback=10):
    swing_high = None
    swing_low = None

    for i in range(lookback, len(df)-lookback):
        # Swing High
        if df["High"].iloc[i] == df["High"].iloc[i-lookback:i+lookback+1].max():
            swing_high = df.index[i]

        # Swing Low
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

# ============================================================
# CHART ENGINE (PRICE + VOLUME + RSI + MACD)
# ============================================================
def plot_ultra_chart(df, bull_divs, bear_divs, hidden_bull, hidden_bear, pullbacks):
    df = df.copy()
    if df.empty:
        return go.Figure()

    # --- AVWAP Anchors ---
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

    # EMA21 + EMA200
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

    # AVWAP High/Low
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

    # Divergences
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

    # Pullbacks
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

# ============================================================
# WEEKLY TREND FILTER (EMA200 + RSI > 50)
# ============================================================
def passes_weekly_trend_filter(ticker):
    df_w = load_data(ticker, interval="1wk", years=1)
    if df_w.empty:
        return False

    df_w = compute_indicators(df_w)

    try:
        return (
            df_w["Close"].iloc[-1] > df_w["EMA200"].iloc[-1] and
            df_w["RSI"].iloc[-1] > 50
        )
    except Exception:
        return False

# ============================================================
# SCREENER ENGINE
# ============================================================
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

# ============================================================
# SPARKLINE GENERATOR
# ============================================================
def make_sparkline(df):
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        line=dict(color="#00e676", width=2)
    ))

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        height=60,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# ============================================================
# WEEKLY TREND STATUS
# ============================================================
def weekly_trend_status(ticker):
    df_w = load_data(ticker, interval="1wk", years=1)
    if df_w.empty:
        return "Unknown"

    df_w = compute_indicators(df_w)

    try:
        if df_w["Close"].iloc[-1] > df_w["EMA200"].iloc[-1] and df_w["RSI"].iloc[-1] > 50:
            return "Bullish"
        else:
            return "Bearish"
    except Exception:
        return "Unknown"

# ============================================================
# WATCHLIST PANEL
# ============================================================
def render_watchlist_panel(watchlist):
    st.sidebar.subheader("📌 Watchlist")

    if not watchlist:
        st.sidebar.info("No tickers added yet.")
        return

    for t in watchlist:
        df = load_data(t, interval="1d", years=1)
        df = compute_indicators(df)

        trend = weekly_trend_status(t)
        color = "🟢" if trend == "Bullish" else "🔴"

        st.sidebar.markdown(f"### {color} {t} — {trend}")

        spark = make_sparkline(df)
        st.sidebar.plotly_chart(spark, use_container_width=True, height=60)

        if st.sidebar.button(f"Open {t} Chart", key=f"open_{t}"):
            st.session_state["chart_ticker"] = t

# ============================================================
# SETTINGS PANEL
# ============================================================
def render_settings_panel():
    st.sidebar.subheader("⚙️ Settings")

    st.session_state["settings"]["default_tf"] = st.sidebar.selectbox(
        "Default Timeframe",
        ["1d", "1h", "15m", "1wk"],
        index=["1d", "1h", "15m", "1wk"].index(st.session_state["settings"]["default_tf"])
    )

    st.session_state["settings"]["default_years"] = st.sidebar.slider(
        "Years of Data",
        1, 5, st.session_state["settings"]["default_years"]
    )

    st.session_state["settings"]["default_risk"] = st.sidebar.number_input(
        "Risk per Trade (₹)",
        min_value=500,
        max_value=20000,
        value=st.session_state["settings"]["default_risk"],
        step=500
    )

    st.sidebar.markdown("---")

    st.session_state["settings"]["show_rsi"] = st.sidebar.checkbox(
        "Show RSI Panel", value=True
    )
    st.session_state["settings"]["show_macd"] = st.sidebar.checkbox(
        "Show MACD Panel", value=True
    )
    st.session_state["settings"]["show_volume"] = st.sidebar.checkbox(
        "Show Volume Panel", value=True
    )
    st.session_state["settings"]["show_ema"] = st.sidebar.checkbox(
        "Show EMA21 + EMA200", value=True
    )
    st.session_state["settings"]["show_avwap"] = st.sidebar.checkbox(
        "Show AVWAP High/Low", value=True
    )
    st.session_state["settings"]["show_divergences"] = st.sidebar.checkbox(
        "Show Divergences", value=True
    )

# ============================================================
# ATR CALCULATION
# ============================================================
def compute_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# ============================================================
# AUTO‑TRADER (SIMULATED EXECUTION)
# ============================================================
def auto_trader(df, signals, risk_per_trade=2000):
    df = df.copy()
    df["ATR"] = compute_atr(df)

    trades = []
    position = None

    for i in range(2, len(df)):
        date = df.index[i]
        close = df["Close"].iloc[i]
        atr = df["ATR"].iloc[i]

        if position is None:
            if date in signals:
                entry_price = df["Open"].iloc[i+1] if i+1 < len(df) else close
                sl = entry_price - 1.5 * atr
                tp = entry_price + 2 * atr

                if atr is None or np.isnan(atr) or atr == 0:
                    continue

                qty = max(1, int(risk_per_trade / max(entry_price - sl, 0.01)))

                position = {
                    "entry_date": date,
                    "entry_price": entry_price,
                    "sl": sl,
                    "tp": tp,
                    "qty": qty
                }
        else:
            low = df["Low"].iloc[i]
            high = df["High"].iloc[i]

            if low <= position["sl"]:
                exit_price = position["sl"]
                pnl = (exit_price - position["entry_price"]) * position["qty"]
                trades.append({
                    "EntryDate": position["entry_date"],
                    "ExitDate": date,
                    "Entry": position["entry_price"],
                    "Exit": exit_price,
                    "PnL": pnl
                })
                position = None
                continue

            if high >= position["tp"]:
                exit_price = position["tp"]
                pnl = (exit_price - position["entry_price"]) * position["qty"]
                trades.append({
                    "EntryDate": position["entry_date"],
                    "ExitDate": date,
                    "Entry": position["entry_price"],
                    "Exit": exit_price,
                    "PnL": pnl
                })
                position = None
                continue

    return pd.DataFrame(trades)

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.set_page_config(page_title="Sniper Divergence Terminal", layout="wide")
    st.title("Sniper Divergence Terminal – AVWAP + RSI + MACD")

    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = []
    if "chart_ticker" not in st.session_state:
        st.session_state["chart_ticker"] = "INFY.NS"

    render_settings_panel()
    render_watchlist_panel(st.session_state["watchlist"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("➕ Manage Watchlist")
    new_ticker = st.sidebar.text_input("Add ticker (e.g. INFY.NS)")
    if st.sidebar.button("Add to Watchlist"):
        t = new_ticker.strip().upper()
        if t and t not in st.session_state["watchlist"]:
            st.session_state["watchlist"].append(t)
    if st.sidebar.button("Clear Watchlist"):
        st.session_state["watchlist"] = []

    col1, col2, col3 = st.columns(3)
    with col1:
        universe = st.selectbox("Universe", ["NIFTY200", "Custom"])
    with col2:
        interval_s = st.selectbox(
            "Timeframe",
            ["1d", "1h", "15m", "1wk"],
            index=["1d", "1h", "15m", "1wk"].index(st.session_state["settings"]["default_tf"])
        )
    with col3:
        fresh_only = st.checkbox("Fresh signals (last 3 candles)", value=True)

    if universe == "Custom":
        custom = st.text_input("Enter tickers (comma separated)", "INFY.NS, TCS.NS")
        tickers = [x.strip().upper() for x in custom.split(",") if x.strip()]
    else:
        tickers = NIFTY200

    use_trend = st.checkbox("Use Weekly Trend Filter (EMA200 + RSI>50)", value=True)

    tab1, tab2, tab3 = st.tabs(["📊 Screener", "📈 Chart", "🤖 Auto‑Trader"])

    # TAB 1: Screener
    with tab1:
        if st.button("Run Screener"):
            df_res = run_screener(tickers, interval_s, fresh_only=fresh_only, use_trend=use_trend)
            if df_res.empty:
                st.warning("No setups found.")
            else:
                st.session_state["last_screener"] = df_res
                st.dataframe(df_res, use_container_width=True)
        else:
            df_res = st.session_state.get("last_screener", pd.DataFrame())
            if not df_res.empty:
                st.dataframe(df_res, use_container_width=True)
            else:
                st.info("Run the screener to see results.")

        df_res = st.session_state.get("last_screener", pd.DataFrame())
        if not df_res.empty:
            st.markdown("### Add from Screener to Watchlist")
            for _, row in df_res.iterrows():
                t = row["Ticker"]
                c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
                with c1:
                    st.write(t)
                with c2:
                    st.write(row["SignalType"])
                with c3:
                    st.write(row["Bias"])
                with c4:
                    if st.button("➕", key=f"add_{t}"):
                        if t not in st.session_state["watchlist"]:
                            st.session_state["watchlist"].append(t)

    # TAB 2: Chart
    with tab2:
        st.markdown("### Chart")

        colc1, colc2 = st.columns([2, 1])
        with colc1:
            ticker = st.text_input("Ticker", st.session_state["chart_ticker"])
        with colc2:
            if st.button("Use from Watchlist Selected?"):
                ticker = st.session_state["chart_ticker"]

        st.session_state["chart_ticker"] = ticker.strip().upper()

        df = load_data(
            st.session_state["chart_ticker"],
            interval=interval_s,
            years=st.session_state["settings"]["default_years"]
        )
        if df.empty:
            st.error("No data for this ticker/timeframe.")
        else:
            df = compute_indicators(df)
            bull, bear, hidden_bull, hidden_bear = compute_divergences(df)
            pullbacks = detect_pullback_setups(df)
            fig = plot_ultra_chart(df, bull, bear, hidden_bull, hidden_bear, pullbacks)
            st.plotly_chart(fig, use_container_width=True)

    # TAB 3: Auto‑Trader
    with tab3:
        st.markdown("### Auto‑Trader (Simulated)")

        ticker_auto = st.text_input("Ticker for Auto‑Trader", st.session_state["chart_ticker"], key="auto_ticker")
        df_auto = load_data(
            ticker_auto,
            interval=interval_s,
            years=st.session_state["settings"]["default_years"]
        )
        if df_auto.empty:
            st.error("No data for this ticker/timeframe.")
        else:
            df_auto = compute_indicators(df_auto)
            bull_a, bear_a, hidden_bull_a, hidden_bear_a = compute_divergences(df_auto)
            pullbacks_a = detect_pullback_setups(df_auto)

            signal_dates = set([d for d, _ in pullbacks_a] + [d for d, _ in bull_a])

            risk = st.session_state["settings"]["default_risk"]
            st.write(f"Risk per trade: ₹{risk}")

            if st.button("Run Auto‑Trader Backtest"):
                trades = auto_trader(df_auto, signal_dates, risk_per_trade=risk)
                if trades.empty:
                    st.warning("No trades generated.")
                else:
                    trades["CumPnL"] = trades["PnL"].cumsum()
                    st.dataframe(trades, use_container_width=True)

                    eq_fig = go.Figure()
                    eq_fig.add_trace(go.Scatter(
                        x=trades["ExitDate"],
                        y=trades["CumPnL"],
                        mode="lines+markers",
                        line=dict(color="#00e676", width=2),
                        name="Equity Curve"
                    ))
                    eq_fig.update_layout(
                        title="Equity Curve",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(eq_fig, use_container_width=True)


# ENTRY POINT
if __name__ == "__main__":
    main()
