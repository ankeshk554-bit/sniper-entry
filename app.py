import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(page_title="Sniper Terminal – Ankesh", layout="wide")

# ============================================================
# CACHING LAYERS (SUPER SPEED)
# ============================================================
@st.cache_data(show_spinner=False)
def load_data(ticker, interval, years=1):
    """Fast cached data loader for all timeframes including weekly."""
    if interval == "1wk":
        period = "5y"   # Y1 selection
        return yf.download(ticker, period=period, interval="1wk",
                           auto_adjust=True, progress=False)
    else:
        return yf.download(ticker, period=f"{years}y", interval=interval,
                           auto_adjust=True, progress=False)

@st.cache_data(show_spinner=False)
def compute_indicators(df):
    """Cached indicator computation."""
    df = df.copy()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    gain_ema = gain.ewm(span=14, adjust=False).mean()
    loss_ema = loss.ewm(span=14, adjust=False).mean()
    rs = gain_ema / loss_ema
    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR
    high, low, close = df["High"], df["Low"], df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=14, adjust=False).mean()

    # AVWAP
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["AVWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

    return df

# ============================================================
# ANCHORED VWAP (CACHED)
# ============================================================
@st.cache_data(show_spinner=False)
def compute_vwaps(df):
    df = df.copy()
    top_idx = df["High"].idxmax()
    bottom_idx = df["Low"].idxmin()

    def anchored(df, anchor_index):
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        vol = df["Volume"]
        cum_vol = vol.iloc[anchor_index:].cumsum()
        cum_tp_vol = (tp * vol).iloc[anchor_index:].cumsum()
        vwap = cum_tp_vol / cum_vol
        full = pd.Series(index=df.index, dtype=float)
        full.iloc[anchor_index:] = vwap
        return full

    df["VWAP_TOP"] = anchored(df, df.index.get_loc(top_idx))
    df["VWAP_BOTTOM"] = anchored(df, df.index.get_loc(bottom_idx))
    return df

# ============================================================
# DIVERGENCE ENGINE (BULLISH + BEARISH)
# ============================================================
def detect_strict_swing_lows(df):
    lows = df["Low"].values
    mask = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)-2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            mask[i] = True
    return mask

def detect_strict_swing_highs(df):
    highs = df["High"].values
    mask = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            mask[i] = True
    return mask

def detect_rsi_bullish_divergence(df, mask):
    divs = []
    idxs = np.where(mask)[0]
    for j in range(1, len(idxs)):
        i1, i2 = idxs[j-1], idxs[j]
        if df["Low"].iloc[i2] < df["Low"].iloc[i1] and df["RSI"].iloc[i2] > df["RSI"].iloc[i1]:
            divs.append((i1, i2))
    return divs

def detect_rsi_bearish_divergence(df, mask):
    divs = []
    idxs = np.where(mask)[0]
    for j in range(1, len(idxs)):
        i1, i2 = idxs[j-1], idxs[j]
        if df["High"].iloc[i2] > df["High"].iloc[i1] and df["RSI"].iloc[i2] < df["RSI"].iloc[i1]:
            divs.append((i1, i2))
    return divs

@st.cache_data(show_spinner=False)
def compute_divergences(df):
    """Cached divergence engine — returns only last bullish + last bearish."""
    lows_mask = detect_strict_swing_lows(df)
    highs_mask = detect_strict_swing_highs(df)

    bull = detect_rsi_bullish_divergence(df, lows_mask)
    bear = detect_rsi_bearish_divergence(df, highs_mask)

    last_bull = [bull[-1]] if bull else []
    last_bear = [bear[-1]] if bear else []

    return last_bull, last_bear
# ============================================================
# WEEKLY TREND FILTER (CACHED)
# ============================================================
@st.cache_data(show_spinner=False)
def get_weekly_trend(ticker):
    df_w = yf.download(
        ticker,
        period="5y",
        interval="1wk",
        auto_adjust=True,
        progress=False
    )
    if df_w.empty:
        return None

    if isinstance(df_w.columns, pd.MultiIndex):
        df_w.columns = df_w.columns.get_level_values(0)

    df_w = df_w.apply(pd.to_numeric, errors="coerce")
    df_w["EMA200"] = df_w["Close"].ewm(span=200, adjust=False).mean()

    # RSI
    delta = df_w["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    gain_ema = gain.ewm(span=14, adjust=False).mean()
    loss_ema = loss.ewm(span=14, adjust=False).mean()
    rs = gain_ema / loss_ema
    df_w["RSI"] = 100 - (100 / (1 + rs))

    df_w = df_w.dropna(subset=["EMA200", "RSI"])
    if df_w.empty:
        return None

    df_w["TrendW"] = (df_w["Close"] > df_w["EMA200"]) & (df_w["RSI"] > 50)
    return df_w["TrendW"]


# ============================================================
# STRENGTH SCORE (uses last bullish divergence)
# ============================================================
def compute_strength(df, i1, i2):
    rsi_slope = df["RSI"].iloc[i2] - df["RSI"].iloc[i1]
    depth = df["Low"].iloc[i1] - df["Low"].iloc[i2]
    ema_dist = df["Close"].iloc[i2] - df["EMA200"].iloc[i2]
    atr_val = df["ATR"].iloc[i2]

    score = (rsi_slope * 0.4) + (depth * 0.3) + (ema_dist * 0.2) + (atr_val * 0.1)
    return round(score, 2)


# ============================================================
# SCREENER ENGINE (FAST + CACHED)
# ============================================================
def scan_stock(ticker, interval, use_trend, fresh_only):
    try:
        df = load_data(ticker, interval, years=1)
        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = compute_indicators(df)
        bull_divs, bear_divs = compute_divergences(df)

        if not bull_divs:
            return None

        # Weekly trend filter
        if use_trend:
            tw = get_weekly_trend(ticker)
            if tw is None:
                return None
            df["TrendW"] = tw.reindex(df.index, method="ffill")

        # Last bullish divergence
        i1, i2 = bull_divs[-1]

        # Fresh divergence check
        if fresh_only and i2 < len(df) - 3:
            return None

        # Trend filter check
        if use_trend and not bool(df["TrendW"].iloc[i2]):
            return None

        # Entry candle check
        next_idx = i2 + 1
        open_n = df["Open"].iloc[next_idx] if next_idx < len(df) else df["Close"].iloc[-1]

        if open_n < df["EMA200"].iloc[i2] or open_n < df["AVWAP"].iloc[i2]:
            return None

        if df["ATR"].iloc[i2] <= 0:
            return None

        strength = compute_strength(df, i1, i2)

        return {
            "Ticker": ticker,
            "SignalDate": df.index[i2],
            "Strength": strength,
            "i1": i1,
            "i2": i2
        }

    except Exception:
        return None


# ============================================================
# ULTRA-PRO PLOTLY CHART (F2 + L1)
# ============================================================
def plot_ultra_pro_chart(df, bull_divs, bear_divs):
    df = compute_vwaps(df)

    fig = go.Figure()

    # -------------------------
    # MAIN PRICE CHART
    # -------------------------
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        yaxis="y1"
    ))

    # EMA200
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA200"],
        mode="lines",
        line=dict(color="orange", width=2),
        name="EMA200",
        yaxis="y1"
    ))

    # AVWAP
    fig.add_trace(go.Scatter(
        x=df.index, y=df["AVWAP"],
        mode="lines",
        line=dict(color="purple", width=2),
        name="AVWAP",
        yaxis="y1"
    ))

    # VWAP FROM TOP
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["VWAP_TOP"],
        mode="lines",
        line=dict(color="red", width=1.5),
        name="VWAP from Top",
        yaxis="y1"
    ))

    # VWAP FROM BOTTOM
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["VWAP_BOTTOM"],
        mode="lines",
        line=dict(color="green", width=1.5),
        name="VWAP from Bottom",
        yaxis="y1"
    ))

    # -------------------------
    # DIVERGENCES (ONLY LAST — F2)
    # -------------------------
    if bull_divs:
        i1, i2 = bull_divs[-1]
        fig.add_trace(go.Scatter(
            x=[df.index[i1], df.index[i2]],
            y=[df["Low"].iloc[i1], df["Low"].iloc[i2]],
            mode="markers+lines",
            marker=dict(color="lime", size=14),   # L1 highlight
            line=dict(color="lime", width=3),     # L1 highlight
            name="Bullish Divergence",
            yaxis="y1"
        ))

    if bear_divs:
        i1_b, i2_b = bear_divs[-1]
        fig.add_trace(go.Scatter(
            x=[df.index[i1_b], df.index[i2_b]],
            y=[df["High"].iloc[i1_b], df["High"].iloc[i2_b]],
            mode="markers+lines",
            marker=dict(color="red", size=14),    # L1 highlight
            line=dict(color="red", width=3),      # L1 highlight
            name="Bearish Divergence",
            yaxis="y1"
        ))

    # -------------------------
    # VOLUME
    # -------------------------
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        marker_color="rgba(0,150,255,0.3)",
        name="Volume",
        yaxis="y2"
    ))

    # -------------------------
    # RSI PANEL
    # -------------------------
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["RSI"],
        mode="lines",
        line=dict(color="cyan", width=2),
        name="RSI",
        yaxis="y3"
    ))

    # RSI divergences (highlight only last)
    if bull_divs:
        i1, i2 = bull_divs[-1]
        fig.add_trace(go.Scatter(
            x=[df.index[i1], df.index[i2]],
            y=[df["RSI"].iloc[i1], df["RSI"].iloc[i2]],
            mode="markers+lines",
            marker=dict(color="lime", size=10),
            line=dict(color="lime", width=2),
            name="Bullish Div (RSI)",
            yaxis="y3"
        ))

    if bear_divs:
        i1_b, i2_b = bear_divs[-1]
        fig.add_trace(go.Scatter(
            x=[df.index[i1_b], df.index[i2_b]],
            y=[df["RSI"].iloc[i1_b], df["RSI"].iloc[i2_b]],
            mode="markers+lines",
            marker=dict(color="red", size=10),
            line=dict(color="red", width=2),
            name="Bearish Div (RSI)",
            yaxis="y3"
        ))

    # -------------------------
    # LAYOUT
    # -------------------------
    fig.update_layout(
        height=950,

        yaxis=dict(title="Price", domain=[0.35, 1.0], side="right"),
        yaxis2=dict(title="Volume", domain=[0.35, 1.0],
                    overlaying="y", side="left", showgrid=False),
        yaxis3=dict(title="RSI", domain=[0, 0.30], range=[0, 100]),

        xaxis=dict(domain=[0, 1], rangeslider=dict(visible=False)),

        showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    return fig


# ============================================================
# BACKTEST ENGINE (FAST)
# ============================================================
def run_backtest(df, bull_divs, risk, trend_series, use_trend):
    df = df.copy()
    df.index.name = "Timestamp"
    df = df.reset_index().reset_index(drop=True)

    if use_trend and trend_series is not None:
        df["TrendW"] = trend_series.reindex(df["Timestamp"], method="ffill")
    elif use_trend:
        return [], 0

    trades = []
    equity = 0

    for (_, i2) in bull_divs:
        if use_trend and not bool(df["TrendW"].iloc[i2]):
            continue

        entry_idx = i2 + 1
        if entry_idx >= len(df):
            continue

        open_n = float(df["Open"].iloc[entry_idx])
        ema_v = float(df["EMA200"].iloc[entry_idx])
        avwap_v = float(df["AVWAP"].iloc[entry_idx])
        atr_v = float(df["ATR"].iloc[entry_idx])

        if atr_v <= 0:
            continue
        if open_n < ema_v or open_n < avwap_v:
            continue

        sl = open_n - 1.5 * atr_v
        tp = open_n + 2 * atr_v
        risk_per_share = open_n - sl

        if risk_per_share <= 0:
            continue

        qty = max(int(risk / risk_per_share), 1)

        exit_p = None
        exit_idx = None

        for j in range(entry_idx + 1, len(df)):
            if df["Low"].iloc[j] <= sl:
                exit_p = sl
                exit_idx = j
                break
            if df["High"].iloc[j] >= tp:
                exit_p = tp
                exit_idx = j
                break

        if exit_p is None:
            exit_p = float(df["Close"].iloc[-1])
            exit_idx = len(df) - 1

        pnl = (exit_p - open_n) * qty
        equity += pnl

        trades.append({
            "Entry": df["Timestamp"].iloc[entry_idx],
            "Exit": df["Timestamp"].iloc[exit_idx],
            "EntryPrice": round(open_n, 2),
            "ExitPrice": round(exit_p, 2),
            "Qty": qty,
            "PnL": round(pnl, 2),
            "Equity": round(equity, 2)
        })

    return trades, equity
# ============================================================
# STREAMLIT UI
# ============================================================
def main():

    tab_screener, tab_backtest = st.tabs(["📊 Screener", "📈 Backtest"])

    # ========================================================
    # 📊 SCREENER TAB
    # ========================================================
    with tab_screener:
        st.title("Sniper Divergence Screener – NIFTY200")

        col1, col2, col3 = st.columns(3)
        with col1:
            universe = st.selectbox("Universe", ["NIFTY200", "Custom"], key="universe_sel")
        with col2:
            interval_s = st.selectbox("Timeframe", ["1d", "1h", "15m", "1wk"], key="screener_tf")
        with col3:
            mode = st.selectbox("View Mode", ["Simple", "Detailed"], key="view_mode_sel")

        use_trend = st.checkbox(
            "Use Weekly Trend Filter (EMA200 + RSI>50)",
            value=True,
            key="trend_filter_screener"
        )

        fresh_only = st.checkbox(
            "Show Only Fresh Divergences (Last 3 Candles)",
            value=True,
            key="fresh_only_toggle"
        )

        # Universe selection
        if universe == "Custom":
            custom = st.text_input("Enter tickers", "HAL.NS", key="custom_tickers")
            tickers = [x.strip() for x in custom.split(",") if x.strip()]
        else:
            tickers = NIFTY200

        # Run Screener Button
        if st.button("Run Screener", key="run_screener_btn"):
            st.info("Scanning…")

            results = []
            for t in tickers:
                r = scan_stock(t, interval_s, use_trend, fresh_only)
                if r:
                    results.append(r)

            if not results:
                st.warning("No setups found.")
            else:
                st.session_state["screener_results"] = pd.DataFrame(results).sort_values("Strength", ascending=False)

        # Display Results + Chart
        if "screener_results" in st.session_state:
            df_res = st.session_state["screener_results"]

            if mode == "Simple":
                st.dataframe(df_res[["Ticker", "SignalDate", "Strength"]], use_container_width=True)
            else:
                st.dataframe(df_res, use_container_width=True)

            selected = st.selectbox(
                "Select a ticker to view chart",
                df_res["Ticker"],
                key="chart_ticker_select"
            )

            if selected:
                row = df_res[df_res["Ticker"] == selected].iloc[0]
                ticker = row["Ticker"]

                df = load_data(ticker, interval_s, years=1)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = compute_indicators(df)
                bull_divs, bear_divs = compute_divergences(df)

                with st.expander(f"📈 Full-Screen Chart – {ticker}", expanded=True):
                    fig = plot_ultra_pro_chart(df, bull_divs, bear_divs)
                    st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # 📈 BACKTEST TAB
    # ========================================================
    with tab_backtest:
        st.title("Sniper Backtester")

        ticker = st.text_input("Ticker", "HAL.NS", key="bt_ticker")
        interval_b = st.selectbox("Timeframe", ["1d", "1h", "15m", "1wk"], key="backtest_tf")
        risk = st.number_input("Risk per Trade (₹)", value=2000, key="bt_risk")
        years = st.slider("Years of Data", 1, 5, 2, key="bt_years")

        use_trend_bt = st.checkbox(
            "Use Weekly Trend Filter (EMA200 + RSI>50)",
            value=True,
            key="trend_filter_backtest"
        )

        if st.button("Run Backtest", key="run_backtest_btn"):
            df = load_data(ticker, interval_b, years=years)

            if df.empty:
                st.error("No data available for this ticker.")
                return

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = compute_indicators(df)
            bull_divs, bear_divs = compute_divergences(df)

            trend_series = get_weekly_trend(ticker) if use_trend_bt else None

            trades, eq = run_backtest(df, bull_divs, risk, trend_series, use_trend_bt)
            df_trades = pd.DataFrame(trades)

            if df_trades.empty:
                st.info("No trades generated.")
            else:
                st.subheader("Backtest Results")
                st.dataframe(df_trades, use_container_width=True)

                st.metric("Total PnL", f"₹{round(eq, 2)}")
                st.metric("Total Trades", len(df_trades))


# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
