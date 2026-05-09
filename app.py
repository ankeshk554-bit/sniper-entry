import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

# ============================================================
# NIFTY200 LIST (TRUNCATED FOR BREVITY — ADD FULL LIST IF NEEDED)
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
# TECHNICAL INDICATORS
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
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cumulative_tp_vol = (typical_price * df['Volume']).cumsum()
    cumulative_vol = df['Volume'].cumsum()
    return cumulative_tp_vol / cumulative_vol

# ============================================================
# DIVERGENCE ENGINE
# ============================================================
def detect_strict_swing_lows(df):
    lows = df['Low'].values
    swing_low = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df) - 2):
        if (
            lows[i] < lows[i-1] and lows[i] < lows[i-2] and
            lows[i] < lows[i+1] and lows[i] < lows[i+2]
        ):
            swing_low[i] = True
    return swing_low

def detect_rsi_bullish_divergence(df, swing_low_mask):
    divergence_points = []
    lows, rsi_vals = df['Low'].values, df['RSI'].values
    swing_indices = np.where(swing_low_mask)[0]
    for i in range(1, len(swing_indices)):
        i1, i2 = swing_indices[i-1], swing_indices[i]
        if lows[i2] < lows[i1] and rsi_vals[i2] > rsi_vals[i1]:
            divergence_points.append((int(i1), int(i2)))
    return divergence_points

def apply_divergence_engine(df):
    df['EMA200'] = ema(df['Close'], 200)
    df['RSI'] = rsi(df['Close'])
    df['ATR'] = atr(df)
    df['AVWAP'] = avwap(df)
    swing_mask = detect_strict_swing_lows(df)
    div_pairs = detect_rsi_bullish_divergence(df, swing_mask)
    return df, div_pairs

# ============================================================
# STRENGTH SCORE
# ============================================================
def compute_strength(df, i1, i2):
    rsi_slope = df['RSI'].iloc[i2] - df['RSI'].iloc[i1]
    price_depth = df['Low'].iloc[i1] - df['Low'].iloc[i2]
    ema_dist = df['Close'].iloc[i2] - df['EMA200'].iloc[i2]
    atr_val = df['ATR'].iloc[i2]

    score = (
        (rsi_slope * 0.4) +
        (price_depth * 0.3) +
        (ema_dist * 0.2) +
        (atr_val * 0.1)
    )
    return round(score, 2)

# ============================================================
# FRESH DIVERGENCE SCREENER
# ============================================================
def scan_stock(ticker, interval):
    try:
        df = yf.download(ticker, period="1y", interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            return None

        df, div_pairs = apply_divergence_engine(df)
        if not div_pairs:
            return None

        i1, i2 = div_pairs[-1]

        # Fresh divergence = last 3 candles
        if i2 < len(df) - 3:
            return None

        # A++ filters
        open_n = df['Open'].iloc[i2+1] if i2+1 < len(df) else df['Close'].iloc[-1]
        if open_n < df['EMA200'].iloc[i2] or open_n < df['AVWAP'].iloc[i2]:
            return None
        if df['ATR'].iloc[i2] <= 0:
            return None

        strength = compute_strength(df, i1, i2)

        return {
            "Ticker": ticker,
            "SignalDate": df.index[i2],
            "RSI": round(df['RSI'].iloc[i2], 2),
            "SwingLow": round(df['Low'].iloc[i2], 2),
            "EMA_Dist": round(df['Close'].iloc[i2] - df['EMA200'].iloc[i2], 2),
            "ATR": round(df['ATR'].iloc[i2], 2),
            "Strength": strength
        }

    except:
        return None

# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(df, div_pairs, risk_per_trade=2000):
    df = df.copy()
    df.index.name = "Timestamp"
    df = df.reset_index().reset_index(drop=True)

    trades, equity = [], 0

    for (_, i2) in div_pairs:
        entry_idx = i2 + 1
        if entry_idx >= len(df):
            continue

        open_n = float(df['Open'].iloc[entry_idx])
        ema_v = float(df['EMA200'].iloc[entry_idx])
        avwap_v = float(df['AVWAP'].iloc[entry_idx])
        atr_v = float(df['ATR'].iloc[entry_idx])

        if any(np.isnan(v) for v in [open_n, ema_v, avwap_v, atr_v]):
            continue
        if atr_v <= 0:
            continue
        if open_n < ema_v or open_n < avwap_v:
            continue

        sl_price = open_n - (1.5 * atr_v)
        tp_price = open_n + (2 * atr_v)

        risk_per_share = open_n - sl_price
        if risk_per_share <= 0:
            continue

        qty = max(int(risk_per_trade / risk_per_share), 1)

        exit_p, exit_idx = None, None
        for j in range(entry_idx + 1, len(df)):
            low_j = float(df['Low'].iloc[j])
            high_j = float(df['High'].iloc[j])

            if low_j <= sl_price:
                exit_p, exit_idx = sl_price, j
                break
            if high_j >= tp_price:
                exit_p, exit_idx = tp_price, j
                break

        if exit_p is None:
            exit_p, exit_idx = float(df['Close'].iloc[-1]), len(df) - 1

        pnl = (exit_p - open_n) * qty
        equity += pnl

        trades.append({
            "Entry": df['Timestamp'].iloc[entry_idx],
            "Exit": df['Timestamp'].iloc[exit_idx],
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
            universe = st.selectbox("Universe", ["NIFTY200", "Custom"], key="universe_key")
        with col2:
            interval = st.selectbox("Timeframe", ["1d", "1h", "15m"], key="screener_tf")
        with col3:
            mode = st.selectbox("View Mode", ["Simple", "Detailed"], key="view_mode")

        if universe == "Custom":
            custom = st.text_input("Enter tickers (comma separated)", "HAL.NS, TCS.NS", key="custom_list")
            tickers = [x.strip() for x in custom.split(",")]
        else:
            tickers = NIFTY200

        if st.button("Run Screener", key="run_screener"):
            st.info("Scanning stocks… please wait")

            results = []
            for t in tickers:
                r = scan_stock(t, interval)
                if r:
                    results.append(r)

            if not results:
                st.warning("No fresh divergence setups found.")
            else:
                df_res = pd.DataFrame(results).sort_values("Strength", ascending=False)

                if mode == "Simple":
                    st.dataframe(df_res[["Ticker", "SignalDate", "Strength"]], use_container_width=True)
                else:
                    st.dataframe(df_res, use_container_width=True)

    # ============================================================
    # BACKTEST TAB
    # ============================================================
    with tab_backtest:
        st.title("Sniper Backtester")

        ticker = st.text_input("Ticker", value="HAL.NS", key="bt_ticker")
        interval = st.selectbox("Timeframe", ["1d", "1h", "15m"], key="backtest_tf")
        risk_per_trade = st.number_input("Risk per Trade (₹)", value=2000, key="bt_risk")
        years = st.slider("Years of Data", 1, 5, 2, key="bt_years")

        if st.button("Run Backtest", key="run_backtest"):
            end_date = date.today()
            start_date = end_date - timedelta(days=365 * years)

            df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=True)
            if df.empty:
                st.error("No data.")
                return

            df, div_pairs = apply_divergence_engine(df)
            trades, _ = run_backtest(df, div_pairs, risk_per_trade)
            trades_df = pd.DataFrame(trades)

            st.dataframe(trades_df, use_container_width=True)

if __name__ == "__main__":
    main()
