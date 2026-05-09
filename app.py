import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import date, timedelta
import plotly.graph_objects as go

# ============================================================
# NIFTY200 LIST
# ============================================================
NIFTY200 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","HINDUNILVR.NS","ITC.NS","LT.NS",
    "SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","HCLTECH.NS","ASIANPAINT.NS","MARUTI.NS","AXISBANK.NS",
    "SUNPHARMA.NS","BAJFINANCE.NS","ULTRACEMCO.NS","WIPRO.NS","DMART.NS","ADANIENT.NS","ADANIPORTS.NS",
    "TITAN.NS","ONGC.NS","POWERGRID.NS","NTPC.NS","JSWSTEEL.NS","TATASTEEL.NS","M&M.NS","BAJAJFINSV.NS",
    "HDFCLIFE.NS","SBILIFE.NS","DIVISLAB.NS","DRREDDY.NS","BRITANNIA.NS","NESTLEIND.NS","HEROMOTOCO.NS",
    "EICHERMOT.NS","BAJAJ-AUTO.NS","COALINDIA.NS","GRASIM.NS","TECHM.NS","CIPLA.NS","SHREECEM.NS",
    "BPCL.NS","IOC.NS","HINDALCO.NS","VEDL.NS","UPL.NS","ABB.NS","AMBUJACEM.NS","APOLLOHOSP.NS",
    "AUROPHARMA.NS","BANDHANBNK.NS","BANKBARODA.NS","BEL.NS","BERGEPAINT.NS","BIOCON.NS","BOSCHLTD.NS",
    "CANBK.NS","CHOLAFIN.NS","CUMMINSIND.NS","DABUR.NS","DLF.NS","GAIL.NS","GLAND.NS","GODREJCP.NS",
    "HAVELLS.NS","ICICIPRULI.NS","IGL.NS","INDHOTEL.NS","INDIGO.NS","INDUSINDBK.NS","JINDALSTEL.NS",
    "LUPIN.NS","MCDOWELL-N.NS","MFSL.NS","MUTHOOTFIN.NS","NAUKRI.NS","PEL.NS","PIDILITIND.NS",
    "PIIND.NS","PNB.NS","POLYCAB.NS","RECLTD.NS","SAIL.NS","SRF.NS","TATACONSUM.NS","TATAMOTORS.NS",
    "TATAPOWER.NS","TORNTPHARM.NS","TRENT.NS","TVSMOTOR.NS","UBL.NS","VOLTAS.NS","ZEEL.NS"
]

# ============================================================
# INDICATORS
# ============================================================
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    gain_ema = gain.ewm(span=length, adjust=False).mean()
    loss_ema = loss.ewm(span=length, adjust=False).mean()
    rs = gain_ema / loss_ema
    return 100 - (100 / (1 + rs))

def atr(df, length=14):
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()

def avwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * df['Volume']).cumsum() / df['Volume'].cumsum()

# ============================================================
# WEEKLY TREND FILTER
# ============================================================
def get_weekly_trend(ticker):
    df_w = yf.download(ticker, period="5y", interval="1wk", auto_adjust=True, progress=False)
    if df_w.empty:
        return None
    if isinstance(df_w.columns, pd.MultiIndex):
        df_w.columns = df_w.columns.get_level_values(0)
    df_w = df_w.apply(pd.to_numeric, errors="coerce")
    df_w["EMA200"] = ema(df_w["Close"], 200)
    df_w["RSI"] = rsi(df_w["Close"])
    df_w = df_w.dropna(subset=["EMA200", "RSI"])
    if df_w.empty:
        return None
    df_w["TrendW"] = (df_w["Close"] > df_w["EMA200"]) & (df_w["RSI"] > 50)
    return df_w["TrendW"]

# ============================================================
# DIVERGENCE ENGINE
# ============================================================
def detect_strict_swing_lows(df):
    lows = df['Low'].values
    mask = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df)-2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            mask[i] = True
    return mask

def detect_rsi_bullish_divergence(df, mask):
    divs = []
    idxs = np.where(mask)[0]
    for i in range(1, len(idxs)):
        i1, i2 = idxs[i-1], idxs[i]
        if df['Low'].iloc[i2] < df['Low'].iloc[i1] and df['RSI'].iloc[i2] > df['RSI'].iloc[i1]:
            divs.append((i1, i2))
    return divs

def apply_divergence_engine(df):
    df = df.copy()
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI"] = rsi(df["Close"])
    df["ATR"] = atr(df)
    df["AVWAP"] = avwap(df)
    mask = detect_strict_swing_lows(df)
    divs = detect_rsi_bullish_divergence(df, mask)
    return df, divs

# ============================================================
# STRENGTH SCORE
# ============================================================
def compute_strength(df, i1, i2):
    rsi_slope = df['RSI'].iloc[i2] - df['RSI'].iloc[i1]
    depth = df['Low'].iloc[i1] - df['Low'].iloc[i2]
    ema_dist = df['Close'].iloc[i2] - df['EMA200'].iloc[i2]
    atr_val = df['ATR'].iloc[i2]
    score = (rsi_slope*0.4) + (depth*0.3) + (ema_dist*0.2) + (atr_val*0.1)
    return round(score, 2)

# ============================================================
# SCREENER
# ============================================================
def scan_stock(ticker, interval, use_trend, fresh_only):
    try:
        df = yf.download(ticker, period="1y", interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df, divs = apply_divergence_engine(df)
        if not divs:
            return None
        if use_trend:
            tw = get_weekly_trend(ticker)
            if tw is None:
                return None
            df["TrendW"] = tw.reindex(df.index, method="ffill")
        i1, i2 = divs[-1]
        if fresh_only and i2 < len(df)-3:
            return None
        if use_trend and not bool(df["TrendW"].iloc[i2]):
            return None
        next_idx = i2 + 1
        open_n = df['Open'].iloc[next_idx] if next_idx < len(df) else df['Close'].iloc[-1]
        if open_n < df['EMA200'].iloc[i2] or open_n < df['AVWAP'].iloc[i2]:
            return None
        if df['ATR'].iloc[i2] <= 0:
            return None
        strength = compute_strength(df, i1, i2)
        return {
            "Ticker": ticker,
            "SignalDate": df.index[i2],
            "Strength": strength,
            "i1": i1,
            "i2": i2
        }
    except:
        return None

# ============================================================
# ULTRA-PRO PLOTLY CHART
# ============================================================
def plot_ultra_pro_chart(df, i1, i2, trend_series):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["EMA200"],
        mode="lines", line=dict(color="orange", width=1.5),
        name="EMA200"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["AVWAP"],
        mode="lines", line=dict(color="purple", width=1.5),
        name="AVWAP"
    ))
    fig.add_trace(go.Scatter(
        x=[df.index[i1], df.index[i2]],
        y=[df["Low"].iloc[i1], df["Low"].iloc[i2]],
        mode="markers+lines",
        marker=dict(color="lime", size=12),
        line=dict(color="lime", width=2),
        name="Bullish Divergence"
    ))
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color="rgba(0,150,255,0.3)",
        name="Volume",
        yaxis="y2"
    ))
    if trend_series is not None:
        trend_series = trend_series.reindex(df.index, method="ffill")
        ribbon_color = ["green" if bool(v) else "gray" for v in trend_series]
        fig.add_trace(go.Bar(
            x=df.index,
            y=[df["Low"].min()*0.0 + 0.0001 for _ in df.index],
            marker_color=ribbon_color,
            name="Weekly Trend",
            yaxis="y3"
        ))
    fig.update_layout(
        height=900,
        xaxis=dict(domain=[0, 1]),
        yaxis=dict(title="Price", side="right"),
        yaxis2=dict(title="Volume", overlaying="y", side="left", showgrid=False, rangemode="tozero"),
        yaxis3=dict(overlaying="y", side="right", showticklabels=False),
        showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(df, divs, risk, trend_series, use_trend):
    df = df.copy()
    df.index.name = "Timestamp"
    df = df.reset_index().reset_index(drop=True)
    if use_trend and trend_series is not None:
        df["TrendW"] = trend_series.reindex(df["Timestamp"], method="ffill")
    elif use_trend:
        return [], 0
    trades, equity = [], 0
    for (_, i2) in divs:
        if use_trend and not bool(df["TrendW"].iloc[i2]):
            continue
        entry_idx = i2 + 1
        if entry_idx >= len(df):
            continue
        open_n = float(df["Open"].iloc[entry_idx])
        ema_v = float(df["EMA200"].iloc[entry_idx])
        avwap_v = float(df["AVWAP"].iloc[entry_idx])
        atr_v = float(df["ATR"].iloc[entry_idx])
        if atr_v <= 0 or open_n < ema_v or open_n < avwap_v:
            continue
        sl = open_n - 1.5 * atr_v
        tp = open_n + 2 * atr_v
        risk_per_share = open_n - sl
        if risk_per_share <= 0:
            continue
        qty = max(int(risk / risk_per_share), 1)
        exit_p, exit_idx = None, None
        for j in range(entry_idx + 1, len(df)):
            if df["Low"].iloc[j] <= sl:
                exit_p, exit_idx = sl, j
                break
            if df["High"].iloc[j] >= tp:
                exit_p, exit_idx = tp, j
                break
        if exit_p is None:
            exit_p, exit_idx = float(df["Close"].iloc[-1]), len(df) - 1
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
    st.set_page_config(page_title="Sniper Terminal – Ankesh", layout="wide")
    tab_screener, tab_backtest = st.tabs(["📊 Screener", "📈 Backtest"])

    # ============================================================
    # SCREENER TAB
    # ============================================================
    with tab_screener:
        st.title("Sniper Divergence Screener – NIFTY200")
        col1, col2, col3 = st.columns(3)
        with col1:
            universe = st.selectbox("Universe", ["NIFTY200", "Custom"])
        with col2:
            interval_s = st.selectbox("Timeframe", ["1d", "1h", "15m"])
        with col3:
            mode = st.selectbox("View Mode", ["Simple", "Detailed"])
        use_trend = st.checkbox("Use Weekly Trend Filter (EMA200 + RSI>50)", value=True)
        fresh_only = st.checkbox("Show Only Fresh Divergences (Last 3 Candles)", value=True)
        if universe == "Custom":
            custom = st.text_input("Enter tickers", "HAL.NS")
            tickers = [x.strip() for x in custom.split(",") if x.strip()]
        else:
            tickers = NIFTY200
        if st.button("Run Screener"):
            st.info("Scanning…")
            results = []
            for t in tickers:
                r = scan_stock(t, interval_s, use_trend, fresh_only)
                if r:
                    results.append(r)
            if not results:
                st.warning("No setups found.")
            else:
                df_res = pd.DataFrame(results).sort_values("Strength", ascending=False)
                if mode == "Simple":
                    st.dataframe(df_res[["Ticker", "SignalDate", "Strength"]], use_container_width=True)
                else:
                    st.dataframe(df_res, use_container_width=True)
                selected = st.selectbox("Select a ticker to view chart", df_res["Ticker"])
                if selected:
                    row = df_res[df_res["Ticker"] == selected].iloc[0]
                    ticker = row["Ticker"]
                    i1, i2 = int(row["i1"]), int(row["i2"])
                    df = yf.download(ticker, period="1y", interval=interval_s, auto_adjust=True, progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df, _ = apply_divergence_engine(df)
                    trend_series = get_weekly_trend(ticker)
                    if trend_series is not None:
                        df["TrendW"] = trend_series.reindex(df.index, method="ffill")
                    with st.expander(f"📈 Full-Screen Chart – {ticker}", expanded=True):
                        fig = plot_ultra_pro_chart(df, i1, i2, df.get("TrendW"))
                        st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # BACKTEST TAB
    # ============================================================
    with tab_backtest:
        with tab_backtest:
    st.title("Sniper Backtester")

    ticker = st.text_input("Ticker", "HAL.NS")
    interval_b = st.selectbox("Timeframe", ["1d", "1h", "15m"])
    risk = st.number_input("Risk per Trade (₹)", value=2000)
    years = st.slider("Years of Data", 1, 5, 2)
    use_trend_bt = st.checkbox("Use Weekly Trend Filter (EMA200 + RSI>50)", value=True)

    if st.button("Run Backtest"):
        end = date.today()
        start = end - timedelta(days=365 * years)
        ...
            end = date.today()
                    if st.button("Run Backtest"):
            end = date.today()
            start = end - timedelta(days=365 * years)

            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval_b,
                auto_adjust=True,
                progress=False
            )

            if df.empty:
                st.error("No data available for this ticker.")
                return

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.apply(pd.to_numeric, errors="coerce")
            df, divs = apply_divergence_engine(df)

            trend_series = get_weekly_trend(ticker) if use_trend_bt else None

            trades, eq = run_backtest(df, divs, risk, trend_series, use_trend_bt)
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

