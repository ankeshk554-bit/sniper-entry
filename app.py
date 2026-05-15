# ============================================================
# IMPORTS
# ============================================================

import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# IMPORT NIFTY SYMBOL LISTS FROM YOUR FILES
# ============================================================

from nifty50 import NIFTY50
from nifty200 import NIFTY200
from nifty500 import NIFTY500

# Convert to .NS format
NIFTY50 = [s + ".NS" for s in NIFTY50]
NIFTY200 = [s + ".NS" for s in NIFTY200]
NIFTY500 = [s + ".NS" for s in NIFTY500]

# ============================================================
# THEME
# ============================================================

def apply_theme():
    st.markdown(
        """
        <style>
        .royal-title {
            font-size: 32px;
            font-weight: 800;
            color: #f5c542;
        }
        .royal-subtitle {
            font-size: 16px;
            font-weight: 500;
            color: #d0d0d0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HELPER INDICATORS (RSI, ATR, OFI, AVWAP)
# ============================================================

def compute_rsi(series: pd.Series, length: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_atr(df: pd.DataFrame, length: int = 14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def compute_ofi(df: pd.DataFrame):
    close = df["Close"]
    vol = df["Volume"]
    prev_close = close.shift(1)

    direction = np.sign(close - prev_close).fillna(0)
    ofi = direction * vol
    return ofi.rolling(5).mean().fillna(0)

def compute_avwap(df: pd.DataFrame):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum()
    cum_pv = (tp * df["Volume"]).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)
# ============================================================
# DATA LOADER WITH RETRY & SAFETY
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(ticker: str, interval: str = "1d", years: int = 3) -> pd.DataFrame:
    for _ in range(3):
        try:
            df = yf.download(
                ticker,
                period=f"{years}y",
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if df is not None and not df.empty:
                df = df.dropna()
                if "Volume" not in df.columns:
                    df["Volume"] = 0
                return df
        except Exception:
            time.sleep(1)
    return pd.DataFrame()


# ============================================================
# CORE INDICATORS ENGINE (WITH MACD)
# ============================================================

def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


@st.cache_data(show_spinner=False)
def compute_indicators(df: pd.DataFrame, swing_bars: int = 5) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # EMAs
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # RSI & ATR
    df["RSI"] = compute_rsi(df["Close"], length=14)
    df["ATR"] = compute_atr(df, length=14)

    # Volume metrics (SAFE)
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    try:
        vol_ratio = (
            df["Volume"].astype(float) /
            df["Vol_MA20"].astype(float)
        ).replace([np.inf, -np.inf], np.nan)
        df["Vol_Ratio"] = vol_ratio.fillna(0.0)
    except Exception:
        df["Vol_Ratio"] = 0.0

    # Order flow
    df["OFI"] = compute_ofi(df)

    # AVWAP
    df["AVWAP"] = compute_avwap(df)

    # MACD (12,26,9)
    macd, macd_signal, macd_hist = compute_macd(df["Close"])
    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd_hist

    # Bollinger Bands
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Mid"] = mid
    df["BB_Upper"] = mid + 2 * std
    df["BB_Lower"] = mid - 2 * std

    # Keltner Channels
    atr20 = df["ATR"].rolling(20).mean()
    df["KC_Mid"] = mid
    df["KC_Upper"] = mid + 1.5 * atr20
    df["KC_Lower"] = mid - 1.5 * atr20

    # Squeeze condition
    df["In_Squeeze"] = (df["BB_Upper"] < df["KC_Upper"]) & (df["BB_Lower"] > df["KC_Lower"])
    df["Squeeze_Fire"] = df["In_Squeeze"].shift(1) & (~df["In_Squeeze"])

    # Divergence flags
    df["Bull_Div"] = False
    df["Bear_Div"] = False

    for i in range(swing_bars, len(df) - swing_bars):
        # Bullish divergence
        if (
            df["Low"].iloc[i] < df["Low"].iloc[i - 1] and
            df["Low"].iloc[i] < df["Low"].iloc[i + 1]
        ):
            prev_idx = i - swing_bars
            if prev_idx >= 0:
                if (
                    df["Low"].iloc[i] < df["Low"].iloc[prev_idx] and
                    df["RSI"].iloc[i] > df["RSI"].iloc[prev_idx]
                ):
                    df.at[df.index[i], "Bull_Div"] = True

        # Bearish divergence
        if (
            df["High"].iloc[i] > df["High"].iloc[i - 1] and
            df["High"].iloc[i] > df["High"].iloc[i + 1]
        ):
            prev_idx = i - swing_bars
            if prev_idx >= 0:
                if (
                    df["High"].iloc[i] > df["High"].iloc[prev_idx] and
                    df["RSI"].iloc[i] < df["RSI"].iloc[prev_idx]
                ):
                    df.at[df.index[i], "Bear_Div"] = True

    return df


# ============================================================
# WEEKLY TREND ENGINE
# ============================================================

@st.cache_data(show_spinner=False)
def get_weekly_trend(ticker: str):
    dfw = load_data(ticker, "1wk", years=5)
    if dfw is None or dfw.empty:
        return None

    c = dfw["Close"]
    dfw["EMA200"] = c.ewm(span=200, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    dfw["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    return (dfw["Close"] > dfw["EMA200"]) & (dfw["RSI"] > 50)
# ============================================================
# VOLUME PROFILE ENGINE
# ============================================================

def compute_volume_profile(df: pd.DataFrame, bins: int = 40):
    if df is None or df.empty:
        return np.array([]), np.array([]), None

    prices = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vols = df["Volume"].astype(float)

    price_min = float(prices.min())
    price_max = float(prices.max())
    if price_min == price_max:
        return np.array([]), np.array([]), None

    edges = np.linspace(price_min, price_max, bins + 1)
    vol_hist = np.zeros(bins)

    idx = np.searchsorted(edges, prices, side="right") - 1
    idx = np.clip(idx, 0, bins - 1)
    for i, v in zip(idx, vols):
        vol_hist[i] += v

    poc_idx = int(np.argmax(vol_hist)) if vol_hist.size > 0 else None
    poc = (edges[poc_idx] + edges[poc_idx + 1]) / 2.0 if poc_idx is not None else None

    return edges, vol_hist, poc


# ============================================================
# CHART ENGINE (PRICE + VOLUME + RSI + MACD)
# ============================================================

def plot_chart(df: pd.DataFrame, ticker: str, show_vp: bool = False):
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{ticker} — No data",
            template=st.session_state.get("plot_theme", "plotly_dark"),
            height=600,
        )
        return fig

    df = df.copy()
    df["Date"] = df.index

    # Base figure with 4 rows: Price, Volume, RSI, MACD
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.45, 0.15, 0.2, 0.2],
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]],
    )

    # ----------------- ROW 1: PRICE + EMAs + AVWAP + DIVERGENCE -----------------
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["EMA21"],
            mode="lines",
            line=dict(color="#fdd835", width=1.2),
            name="EMA21",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["EMA50"],
            mode="lines",
            line=dict(color="#42a5f5", width=1.2),
            name="EMA50",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["EMA200"],
            mode="lines",
            line=dict(color="#ab47bc", width=1.2),
            name="EMA200",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["AVWAP"],
            mode="lines",
            line=dict(color="#ffb300", width=1.2, dash="dot"),
            name="AVWAP",
        ),
        row=1,
        col=1,
    )

    # Divergence markers
    bull_idx = df.index[df["Bull_Div"] == True]
    bear_idx = df.index[df["Bear_Div"] == True]

    if len(bull_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=df.loc[bull_idx, "Date"],
                y=df.loc[bull_idx, "Low"] * 0.995,
                mode="markers",
                marker=dict(color="#00e676", size=9, symbol="triangle-up"),
                name="Bull Div",
            ),
            row=1,
            col=1,
        )

    if len(bear_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=df.loc[bear_idx, "Date"],
                y=df.loc[bear_idx, "High"] * 1.005,
                mode="markers",
                marker=dict(color="#ff1744", size=9, symbol="triangle-down"),
                name="Bear Div",
            ),
            row=1,
            col=1,
        )

    # Optional Volume Profile on price axis
    if show_vp:
        edges, vol_hist, poc = compute_volume_profile(df)
        if vol_hist.size > 0:
            vol_norm = vol_hist / vol_hist.max()
            price_levels = (edges[:-1] + edges[1:]) / 2.0
            price_range = df["Close"].max() - df["Close"].min()
            width = price_range * 0.15

            fig.add_trace(
                go.Bar(
                    x=[df["Date"].iloc[-1]] * len(price_levels),
                    y=price_levels,
                    orientation="h",
                    width=width * vol_norm,
                    marker=dict(color="rgba(100, 181, 246, 0.4)"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    # ----------------- ROW 2: VOLUME -----------------
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Volume"],
            marker_color="#42a5f5",
            name="Volume",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Vol_MA20"],
            mode="lines",
            line=dict(color="#fdd835", width=1.0),
            name="Vol MA20",
        ),
        row=2,
        col=1,
    )

    # ----------------- ROW 3: RSI -----------------
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["RSI"],
            mode="lines",
            line=dict(color="#26c6da", width=1.5),
            name="RSI",
        ),
        row=3,
        col=1,
    )

    fig.add_hline(y=70, line=dict(color="#ef5350", width=1, dash="dot"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="#66bb6a", width=1, dash="dot"), row=3, col=1)

    # ----------------- ROW 4: MACD -----------------
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MACD"],
            mode="lines",
            line=dict(color="#26a69a", width=1.5),
            name="MACD",
        ),
        row=4,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MACD_Signal"],
            mode="lines",
            line=dict(color="#ffca28", width=1.2),
            name="Signal",
        ),
        row=4,
        col=1,
    )

    # MACD Histogram (green/red)
    hist_colors = np.where(df["MACD_Hist"] >= 0, "#00e676", "#ff1744")
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["MACD_Hist"],
            marker_color=hist_colors,
            name="MACD Hist",
        ),
        row=4,
        col=1,
    )

    # ----------------- LAYOUT -----------------
    fig.update_layout(
        title=f"{ticker} — Sniper Terminal v4",
        template=st.session_state.get("plot_theme", "plotly_dark"),
        xaxis_rangeslider_visible=False,
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig.update_yaxes(showgrid=True, zeroline=False, row=1, col=1)
    fig.update_yaxes(showgrid=True, zeroline=False, row=2, col=1)
    fig.update_yaxes(showgrid=True, zeroline=False, row=3, col=1)
    fig.update_yaxes(showgrid=True, zeroline=False, row=4, col=1)

    return fig
# ============================================================
# BACKTEST — SIGNAL GENERATOR (NON‑REPAINTING)
# ============================================================

def generate_signals_for_backtest(
    df: pd.DataFrame,
    ticker: str,
    setup_type: str,
    total_capital: float,
    risk_pct: float,
):
    signals = []
    if df is None or df.empty:
        return signals

    for i2 in range(30, len(df) - 2):

        atr = float(df["ATR"].iloc[i2])
        if atr <= 0 or np.isnan(atr):
            continue

        ei = i2 + 1
        entry = float(df["Open"].iloc[ei])

        q = signal_quality(df, i2)
        if q["total"] < 40:
            continue

        # -----------------------------
        # SETUP‑SPECIFIC LOGIC
        # -----------------------------
        if setup_type == "Divergence":
            if not (df["Bull_Div"].iloc[i2] or df["Bear_Div"].iloc[i2]):
                continue

            if df["Bull_Div"].iloc[i2]:
                sl = round(float(df["Low"].iloc[i2]) - 0.5 * atr, 2)
                tp = round(entry + 2.0 * atr, 2)
            else:
                sl = round(float(df["High"].iloc[i2]) + 0.5 * atr, 2)
                tp = round(entry - 2.0 * atr, 2)

        elif setup_type == "BB Squeeze":
            if not bool(df["Squeeze_Fire"].iloc[i2]):
                continue
            sl = round(float(df["Low"].iloc[i2]) - 0.5 * atr, 2)
            tp = round(entry + 2.0 * atr, 2)

        elif setup_type == "High Volume Breakout":
            vol = df["Volume"]
            vol_ma20 = vol.rolling(20).mean()
            if vol_ma20.iloc[i2] <= 0:
                continue
            if vol.iloc[i2] / vol_ma20.iloc[i2] < 2.0:
                continue
            sl = round(float(df["Close"].iloc[i2]) - 1.0 * atr, 2)
            tp = round(entry + 2.0 * atr, 2)

        elif setup_type == "Trend Pullback":
            ema21 = float(df["EMA21"].iloc[i2])
            ema50 = float(df["EMA50"].iloc[i2])
            ema200 = float(df["EMA200"].iloc[i2])
            close_i2 = float(df["Close"].iloc[i2])

            if not (close_i2 > ema21 and close_i2 > ema50 and close_i2 > ema200):
                continue
            if not detect_reversal_candle(df, i2):
                continue

            sl = round(float(df["Low"].iloc[i2]) - 0.5 * atr, 2)
            tp = round(entry + 2.0 * atr, 2)

        elif setup_type == "Liquidity Sweep":
            if not is_stop_hunt_wick(df, i2, side="bull"):
                continue
            sl = round(float(df["Low"].iloc[i2]) - 0.3 * atr, 2)
            tp = round(entry + 2.5 * atr, 2)

        elif setup_type == "VCP Pattern":
            if float(df["Vol_Ratio"].iloc[i2]) < 1.5:
                continue
            if float(df["Close"].iloc[i2]) < float(df["EMA200"].iloc[i2]):
                continue
            sl = round(float(df["Close"].iloc[i2]) - 1.0 * atr, 2)
            tp = round(entry + 2.5 * atr, 2)

        else:
            continue

        # -----------------------------
        # RISK MANAGEMENT
        # -----------------------------
        sl_dist = max(abs(entry - sl), 0.01)
        risk_amount = total_capital * risk_pct / 100.0
        qty = max(int(risk_amount / sl_dist), 1)

        rr = round((tp - entry) / sl_dist, 2)

        signals.append(
            {
                "Ticker": ticker,
                "Date": str(df.index[i2])[:10],
                "Entry": entry,
                "SL": sl,
                "TP": tp,
                "i2": i2,
                "Qty": qty,
                "R_R": rr,
                "Grade": q["grade"],
            }
        )

    return signals


# ============================================================
# BACKTEST — TRADE SIMULATION
# ============================================================

def simulate_trade(df: pd.DataFrame, sig: dict):
    i2 = sig["i2"]
    entry = sig["Entry"]
    sl = sig["SL"]
    tp = sig["TP"]
    qty = sig["Qty"]

    # Max 25 bars in trade
    for j in range(i2 + 1, min(i2 + 25, len(df))):
        low = float(df["Low"].iloc[j])
        high = float(df["High"].iloc[j])

        # LONG ONLY (Sniper Terminal v4 is long‑biased)
        if low <= sl:
            return "SL", sl, (sl - entry) * qty

        if high >= tp:
            return "TP", tp, (tp - entry) * qty

    # Exit at last bar close
    exit_price = float(df["Close"].iloc[min(i2 + 25, len(df) - 1)])
    return "EXIT", exit_price, (exit_price - entry) * qty


# ============================================================
# BACKTEST — ENGINE
# ============================================================

def backtest_setup(df: pd.DataFrame, signals: list, starting_capital: float):
    if not signals:
        return pd.DataFrame()

    capital = starting_capital
    trades = []

    for sig in signals:
        outcome, exit_price, pnl = simulate_trade(df, sig)
        capital += pnl

        trades.append(
            {
                "Date": sig["Date"],
                "Ticker": sig["Ticker"],
                "Entry": sig["Entry"],
                "Exit": round(exit_price, 2),
                "Hit": outcome,
                "PnL": round(pnl, 2),
                "Capital": round(capital, 2),
                "R_R": sig["R_R"],
                "Grade": sig["Grade"],
            }
        )

    return pd.DataFrame(trades)
# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

def sidebar_navigation():
    st.sidebar.markdown(
        "<div class='royal-title'>Sniper Terminal v4</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<div class='royal-subtitle'>Royal Gold Edition</div><br>",
        unsafe_allow_html=True,
    )

    nav = st.sidebar.radio(
        "Navigation",
        ["Screener", "Chart", "Backtest", "Autotrade", "Volume Profile"],
        index=["Screener", "Chart", "Backtest", "Autotrade", "Volume Profile"].index(
            st.session_state.get("nav_page", "Screener")
        ),
    )

    st.session_state["nav_page"] = nav
    return nav


# ============================================================
# SETUP SELECTOR
# ============================================================

def setup_selector():
    setups = [
        "Divergence",
        "BB Squeeze",
        "High Volume Breakout",
        "Trend Pullback",
        "Liquidity Sweep",
        "VCP Pattern",
    ]

    selected = st.segmented_control(
        "Select Setup",
        setups,
        default=st.session_state.get("selected_setup", "Divergence"),
    )

    st.session_state["selected_setup"] = selected
    return selected


# ============================================================
# SCREENER PAGE
# ============================================================

def page_screener():
    st.markdown("<div class='royal-title'>Screener</div>", unsafe_allow_html=True)

    setup_type = setup_selector()

    col1, col2, col3 = st.columns(3)
    with col1:
        interval = st.selectbox("Interval", ["1d", "1h", "15m"])
    with col2:
        use_trend = st.checkbox("Weekly Trend Filter", True)
    with col3:
        fresh_only = st.checkbox("Fresh Signals Only", True)

    col4, col5, col6 = st.columns(3)
    with col4:
        swing_bars = st.number_input("Swing Bars", 3, 10, 5)
    with col5:
        min_q = st.number_input("Min Quality", 0, 100, 50)
    with col6:
        total_capital = st.number_input("Total Capital", 10000, 10000000, 200000)

    risk_pct = st.slider("Risk % per Trade", 0.1, 5.0, 1.0)

    universe = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
        "SBIN.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "HINDUNILVR.NS",
        "MARUTI.NS", "TITAN.NS", "BAJFINANCE.NS", "ITC.NS", "ULTRACEMCO.NS",
    ]

    if st.button("Run Screener"):
        rows = []
        for ticker in universe:
            df = load_data(ticker, interval)
            if df.empty:
                continue
            df = compute_indicators(df, swing_bars=swing_bars)

            weekly_trend = None
            if use_trend:
                wt = get_weekly_trend(ticker)
                if wt is not None and len(wt) > 0:
                    weekly_trend = bool(wt.iloc[-1])

            sigs = generate_signals_for_backtest(
                df,
                ticker,
                setup_type,
                total_capital,
                risk_pct,
            )
            if not sigs:
                continue

            last_sig = sigs[-1]
            if fresh_only:
                # only if signal is within last 3 bars
                if df.index[-1] != df.index[last_sig["i2"]] and df.index[-2] != df.index[last_sig["i2"]] and df.index[-3] != df.index[last_sig["i2"]]:
                    continue

            row = {
                "Ticker": ticker,
                "Date": last_sig["Date"],
                "Entry": last_sig["Entry"],
                "SL": last_sig["SL"],
                "TP": last_sig["TP"],
                "R_R": last_sig["R_R"],
                "Grade": last_sig["Grade"],
                "Weekly_Trend": weekly_trend,
                "i2": last_sig["i2"],
            }
            rows.append(row)

        if not rows:
            st.warning("No signals found.")
            return

        df_res = pd.DataFrame(rows)
        st.dataframe(df_res, use_container_width=True)

        st.markdown("---")
        st.subheader("Setup Analysis")

        selected_row = st.selectbox("Select a signal", df_res.index)
        row = df_res.loc[selected_row]

        ticker = row["Ticker"]
        df_full = load_data(ticker, interval)
        df_full = compute_indicators(df_full)

        i2 = int(row["i2"])
        st.write(f"Signal index: {i2}, Date: {df_full.index[i2]}")

        st.write("Entry:", row["Entry"], "SL:", row["SL"], "TP:", row["TP"], "R:R:", row["R_R"])


# ============================================================
# CHART PAGE
# ============================================================

def page_chart():
    st.markdown("<div class='royal-title'>Chart</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Ticker", "RELIANCE.NS")
    with col2:
        tf = st.selectbox("Timeframe", ["Daily", "Weekly", "1h", "15m"])
    with col3:
        show_vp = st.checkbox("Show Volume Profile", False)

    if tf == "Daily":
        interval = "1d"
    elif tf == "Weekly":
        interval = "1wk"
    elif tf == "1h":
        interval = "1h"
    else:
        interval = "15m"

    if st.button("Load Chart"):
        df = load_data(ticker, interval)
        if df.empty:
            st.error("No data for this symbol/interval.")
            return

        df = compute_indicators(df)
        fig = plot_chart(df, ticker, show_vp=show_vp)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# BACKTEST PAGE
# ============================================================

def page_backtest():
    st.markdown("<div class='royal-title'>Backtest</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Ticker", "RELIANCE.NS")
    with col2:
        interval = st.selectbox("Interval", ["1d", "1h"])
    with col3:
        setup_type = setup_selector()

    col4, col5 = st.columns(2)
    with col4:
        total_capital = st.number_input("Starting Capital", 10000, 10000000, 200000)
    with col5:
        risk_pct = st.slider("Risk % per Trade", 0.1, 5.0, 1.0)

    if st.button("Run Backtest"):
        df = load_data(ticker, interval)
        if df.empty:
            st.error("No data for this symbol/interval.")
            return

        df = compute_indicators(df)

        signals = generate_signals_for_backtest(
            df,
            ticker,
            setup_type,
            total_capital,
            risk_pct,
        )

        if not signals:
            st.warning("No signals found for backtest.")
            return

        bt = backtest_setup(df, signals, total_capital)
        if bt.empty:
            st.warning("Backtest produced no trades.")
            return

        st.subheader("Trades")
        st.dataframe(bt, use_container_width=True)

        st.subheader("Equity Curve")
        st.line_chart(bt.set_index("Date")["Capital"])

        wins = (bt["PnL"] > 0).sum()
        losses = (bt["PnL"] <= 0).sum()
        winrate = wins / max(wins + losses, 1) * 100
        avg_r = (bt["PnL"] / (bt["PnL"].abs() + 1e-6)).mean()

        st.markdown(f"**Winrate:** {winrate:.1f}%")
        st.markdown(f"**Trades:** {len(bt)}")


# ============================================================
# AUTOTRADE PAGE (PAPER)
# ============================================================

def page_autotrade():
    st.markdown("<div class='royal-title'>Autotrade (Paper)</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Ticker", "RELIANCE.NS")
    with col2:
        interval = st.selectbox("Interval", ["1d", "1h", "15m"])
    with col3:
        setup_type = setup_selector()

    col4, col5 = st.columns(2)
    with col4:
        total_capital = st.number_input("Capital", 10000, 10000000, 200000)
    with col5:
        risk_pct = st.slider("Risk % per Trade", 0.1, 5.0, 1.0)

    if st.button("Execute Last Signal"):
        df = load_data(ticker, interval)
        if df.empty:
            st.error("No data for this symbol/interval.")
            return

        df = compute_indicators(df)

        signals = generate_signals_for_backtest(
            df,
            ticker,
            setup_type,
            total_capital,
            risk_pct,
        )

        if not signals:
            st.warning("No valid signal found.")
            return

        sig = signals[-1]
        outcome, exit_price, pnl = simulate_trade(df, sig)

        trade = {
            "Ticker": ticker,
            "Date": sig["Date"],
            "Entry": sig["Entry"],
            "SL": sig["SL"],
            "TP": sig["TP"],
            "Qty": sig["Qty"],
            "Outcome": outcome,
            "Exit_Price": round(exit_price, 2),
            "PnL": round(pnl, 2),
        }

        st.json(trade)


# ============================================================
# VOLUME PROFILE PAGE
# ============================================================

def page_volume_profile():
    st.markdown("<div class='royal-title'>Volume Profile</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker", "RELIANCE.NS")
    with col2:
        interval = st.selectbox("Interval", ["1d", "1h"])

    if st.button("Compute Volume Profile"):
        df = load_data(ticker, interval)
        if df.empty:
            st.error("No data for this symbol/interval.")
            return

        df = compute_indicators(df)
        edges, vol_hist, poc = compute_volume_profile(df)

        if vol_hist.size == 0:
            st.warning("Not enough data to compute volume profile.")
            return

        st.write("POC:", poc)

        vp_df = pd.DataFrame(
            {
                "Price": (edges[:-1] + edges[1:]) / 2.0,
                "Volume": vol_hist,
            }
        ).set_index("Price")

        st.bar_chart(vp_df["Volume"])


# ============================================================
# MAIN APP
# ============================================================

def main():
    st.set_page_config(
        page_title="Sniper Terminal v4 — Royal Gold",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "plot_theme" not in st.session_state:
        st.session_state["plot_theme"] = "plotly_dark"

    nav = sidebar_navigation()

    if nav == "Screener":
        page_screener()
    elif nav == "Chart":
        page_chart()
    elif nav == "Backtest":
        page_backtest()
    elif nav == "Autotrade":
        page_autotrade()
    elif nav == "Volume Profile":
        page_volume_profile()


if __name__ == "__main__":
    main()
