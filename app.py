import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

# ============================
# 1. TECHNICAL INDICATORS
# ============================
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

# ============================
# 2. DIVERGENCE ENGINE
# ============================
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
    df['SwingLow'] = swing_mask
    div_pairs = detect_rsi_bullish_divergence(df, swing_mask)
    
    df['Div_Arrow'] = np.nan
    df['Div_Line_Price'] = np.nan
    df['Div_Line_RSI'] = np.nan
    for (i1, i2) in div_pairs:
        df.iloc[i2, df.columns.get_loc('Div_Arrow')] = df['Low'].iloc[i2] * 0.995
        df.iloc[i1:i2+1, df.columns.get_loc('Div_Line_Price')] = np.linspace(
            df['Low'].iloc[i1], df['Low'].iloc[i2], i2 - i1 + 1
        )
        df.iloc[i1:i2+1, df.columns.get_loc('Div_Line_RSI')] = np.linspace(
            df['RSI'].iloc[i1], df['RSI'].iloc[i2], i2 - i1 + 1
        )
    return df, div_pairs

# ============================
# 3. BACKTEST ENGINE
# ============================
def run_backtest(df, divergence_pairs, risk_per_trade=2000):
    df = df.copy()
    df = df.reset_index().rename(columns={"index": "Timestamp"})
    df = df.reset_index(drop=True)

    trades, equity = [], 0

    for (_, i2) in divergence_pairs:
        entry_idx = i2 + 1
        if entry_idx >= len(df):
            continue

        open_n = float(df['Open'].iloc[entry_idx])
        ema_v = float(df['EMA200'].iloc[entry_idx])
        avwap_v = float(df['AVWAP'].iloc[entry_idx])
        atr_v = float(df['ATR'].iloc[entry_idx])

        if np.isnan(open_n) or np.isnan(ema_v) or np.isnan(avwap_v) or np.isnan(atr_v):
            continue
        if atr_v <= 0:
            continue
        if open_n < ema_v or open_n < avwap_v:
            continue

        sl_price = open_n - (1.5 * atr_v)
        tp_price = open_n + (2 * atr_v)

        if open_n < sl_price:
            continue

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
            "EntryPrice": open_n,
            "ExitPrice": exit_p,
            "Qty": qty,
            "PnL": pnl,
            "Equity": equity
        })
    return trades, equity

# ============================
# 4. STREAMLIT UI
# ============================
def main():
    st.set_page_config(page_title="Sniper Terminal – Ankesh", layout="wide")
    st.title("Sniper Terminal – Ankesh")
    st.caption("Strict RSI Bullish Divergence • Trend + AVWAP + ATR Engine")

    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker", value="HAL.NS")
        interval = st.selectbox("Timeframe", ["1d", "1h", "15m"], index=0)
        risk_per_trade = st.number_input("Risk per Trade (₹)", value=2000, min_value=100, step=100)
        years = st.slider("Years of Data", 1, 5, 2)
        run_btn = st.button("Run Backtest", type="primary")

    if run_btn:
        end_date = date.today()
        start_date = end_date - timedelta(days=365 * years)
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, auto_adjust=True)

        if df.empty:
            st.error("No data loaded. Check ticker or timeframe.")
            return

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df, div_pairs = apply_divergence_engine(df)
        trades, _ = run_backtest(df, div_pairs, risk_per_trade)
        trades_df = pd.DataFrame(trades)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name="Price"
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], name='EMA200',
                                 line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['AVWAP'], name='AVWAP',
                                 line=dict(color='purple', width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Div_Line_Price'], name='Bull Div',
                                 line=dict(color='lime', width=1, dash='dot')))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Div_Arrow'], mode='markers', name='Div Arrow',
            marker=dict(symbol='triangle-up', size=10, color='lime')
        ))

        fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        if not trades_df.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Win Rate", f"{(trades_df['PnL'] > 0).mean() * 100:.1f}%")
            c2.metric("Net PnL", f"₹{trades_df['PnL'].sum():,.0f}")
            c3.metric("Total Trades", len(trades_df))
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No trades found matching A++ criteria in this period.")

if __name__ == "__main__":
    main()
