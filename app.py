import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

# ============================
# 1. EXPONENTIAL MOVING AVERAGE
# ============================
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

# ============================
# 2. RELATIVE STRENGTH INDEX
# ============================
def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    gain_ema = gain.ewm(span=length, adjust=False).mean()
    loss_ema = loss.ewm(span=length, adjust=False).mean()
    rs = gain_ema / loss_ema
    return 100 - (100 / (1 + rs))

# ============================
# 3. AVERAGE TRUE RANGE (ATR)
# ============================
def atr(df, length=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()

# ============================
# 4. ANCHORED VWAP (AVWAP)
# ============================
def avwap(df):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cumulative_tp_vol = (typical_price * df['Volume']).cumsum()
    cumulative_vol = df['Volume'].cumsum()
    return cumulative_tp_vol / cumulative_vol

# ============================
# 5. STRICT SWING-LOW DETECTOR
# ============================
def detect_strict_swing_lows(df):
    lows = df['Low'].values
    swing_low = np.zeros(len(df), dtype=bool)
    for i in range(2, len(df) - 2):
        if (
            lows[i] < lows[i - 1] and
            lows[i] < lows[i - 2] and
            lows[i] < lows[i + 1] and
            lows[i] < lows[i + 2]
        ):
            swing_low[i] = True
    return swing_low

# ============================
# 6. STRICT RSI BULLISH DIVERGENCE
# ============================
def detect_rsi_bullish_divergence(df, swing_low_mask):
    divergence_points = []
    lows = df['Low'].values
    rsi_vals = df['RSI'].values
    swing_indices = np.where(swing_low_mask)[0]

    for i in range(1, len(swing_indices)):
        i1 = swing_indices[i - 1]
        i2 = swing_indices[i]
        if lows[i2] < lows[i1] and rsi_vals[i2] > rsi_vals[i1]:
            divergence_points.append((int(i1), int(i2)))

    return divergence_points

# ============================
# 7. DIVERGENCE MARKER GENERATOR
# ============================
def generate_divergence_markers(df, divergence_pairs):
    df['Div_Arrow'] = np.nan
    df['Div_Line_Price'] = np.nan
    df['Div_Line_RSI'] = np.nan

    for (i1, i2) in divergence_pairs:
        df.iloc[i2, df.columns.get_loc('Div_Arrow')] = df['Low'].iloc[i2] * 0.995

        df.iloc[i1:i2+1, df.columns.get_loc('Div_Line_Price')] = np.linspace(
            df['Low'].iloc[i1],
            df['Low'].iloc[i2],
            i2 - i1 + 1
        )

        df.iloc[i1:i2+1, df.columns.get_loc('Div_Line_RSI')] = np.linspace(
            df['RSI'].iloc[i1],
            df['RSI'].iloc[i2],
            i2 - i1 + 1
        )

    return df

# ============================
# 8. FULL DIVERGENCE ENGINE
# ============================
def apply_divergence_engine(df):
    df['EMA200'] = ema(df['Close'], 200)
    df['RSI'] = rsi(df['Close'])
    df['ATR'] = atr(df)
    df['AVWAP'] = avwap(df)

    swing_mask = detect_strict_swing_lows(df)
    df['SwingLow'] = swing_mask

    div_pairs = detect_rsi_bullish_divergence(df, swing_mask)
    df = generate_divergence_markers(df, div_pairs)

    return df, div_pairs

# ============================
# 9. CLEAN BACKTEST ENGINE (FINAL FIX)
# ============================
def run_backtest(df, divergence_pairs, risk_per_trade=2000):
    df = df.copy()
    df = df.reset_index(drop=False)   # <-- FIX: force RangeIndex
    df = df.reset_index(drop=True)    # <-- FIX: clean 0..N index

    trades = []
    equity = 0

    for (_, i2) in divergence_pairs:
        entry_index = i2 + 1
        if entry_index >= len(df):
            continue

        open_next = float(df['Open'].iloc[entry_index])
        ema_val = float(df['EMA200'].iloc[entry_index])
        avwap_val = float(df['AVWAP'].iloc[entry_index])
        atr_val = float(df['ATR'].iloc[entry_index])

        if atr_val <= 0:
            continue
        if open_next < ema_val:
            continue
        if open_next < avwap_val:
            continue

        sl_distance = 1.5 * atr_val
        sl_price = open_next - sl_distance

        if open_next < sl_price:
            continue

        qty = max(int(risk_per_trade / sl_distance), 1)
        tp_price = open_next + 2 * atr_val

        exit_price = None
        exit_index = None

        for j in range(entry_index + 1, len(df)):
            low_j = float(df['Low'].iloc[j])
            high_j = float(df['High'].iloc[j])

            if low_j <= sl_price:
                exit_price = sl_price
                exit_index = j
                break

            if high_j >= tp_price:
                exit_price = tp_price
                exit_index = j
                break

        if exit_price is None:
            exit_price = float(df['Close'].iloc[-1])
            exit_index = len(df) - 1

        pnl = (exit_price - open_next) * qty
        equity += pnl

        trades.append({
            "Entry": df['index'].iloc[entry_index],
            "Exit": df['index'].iloc[exit_index],
            "EntryPrice": open_next,
            "ExitPrice": exit_price,
            "Qty": qty,
            "PnL": pnl,
            "Equity": equity
        })

    return trades, equity

# ============================
# 10. TRADES → DF
# ============================
def trades_to_df(trades):
    if len(trades) == 0:
        return pd.DataFrame(columns=[
            "Entry", "Exit", "EntryPrice", "ExitPrice", "Qty", "PnL", "Equity"
        ])
    return pd.DataFrame(trades)

def build_equity_curve(trades_df):
    if len(trades_df) == 0:
        return pd.Series(dtype=float)
    return trades_df["Equity"]

# ============================
# 11. STREAMLIT APP
# ============================
@st.cache_data
def load_data(ticker, start, end, interval):
    df = yf.download(ticker, start=start, end=end, interval=interval)
    df.dropna(inplace=True)
    return df

def plot_chart(df, trades_df):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA200'],
        mode='lines',
        name='EMA200',
        line=dict(color='orange', width=1)
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['AVWAP'],
        mode='lines',
        name='AVWAP',
        line=dict(color='purple', width=1)
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Div_Line_Price'],
        mode='lines',
        name='Div Price Line',
        line=dict(color='lime', width=1, dash='dot'),
        connectgaps=False
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Div_Arrow'],
        mode='markers',
        name='Bullish Div',
        marker=dict(symbol='triangle-up', size=10, color='lime'),
    ))

    if len(trades_df) > 0:
        fig.add_trace(go.Scatter(
            x=trades_df['Entry'],
            y=trades_df['EntryPrice'],
            mode='markers',
            name='Entry',
            marker=dict(symbol='circle', size=9, color='cyan')
        ))
        fig.add_trace(go.Scatter(
            x=trades_df['Exit'],
            y=trades_df['ExitPrice'],
            mode='markers',
            name='Exit',
            marker=dict(symbol='x', size=9, color='red')
        ))

    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        title="Price with RSI Bullish Divergence & Trades"
    )

    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='white', width=1)
    ))
    rsi_fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Div_Line_RSI'],
        mode='lines',
        name='Div RSI Line',
        line=dict(color='lime', width=1, dash='dot'),
        connectgaps=False
    ))
    rsi_fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.1, line_width=0)
    rsi_fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        title="RSI + Divergence"
    )

    atr_fig = go.Figure()
    atr_fig.add_trace(go.Scatter(
        x=df.index,
        y=df['ATR'],
        mode='lines',
        name='ATR',
        line=dict(color='yellow', width=1)
    ))
    atr_fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        title="ATR"
    )

    return fig, rsi_fig, atr_fig

def compute_metrics(trades_df):
    if len(trades_df) == 0:
        return 0, 0.0, 0.0, 0.0

    total_trades = len(trades_df)
    wins = (trades_df['PnL'] > 0).sum()
    win_rate = (wins / total_trades) * 100

    net_pnl = trades_df['PnL'].sum()
    equity_curve = trades_df['Equity']

    max_dd = 0.0
    if len(equity_curve) > 0:
        peak = equity_curve.iloc[0]
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd

    return total_trades, win_rate, net_pnl, max_dd

def main():
    st.set_page_config(page_title="Sniper Terminal – Ankesh", layout="wide")

    st.title("Sniper Terminal – Ankesh")
    st.caption("Strict RSI Bullish Divergence • Trend + AVWAP + ATR Engine")

    with st.sidebar:
        st.header("Settings")

        ticker = st.text_input("Ticker", value="HAL.NS")
        interval = st.selectbox("Timeframe", ["1d", "1h", "15m"], index=0)
        risk_per_trade = st.number_input("Risk per Trade (₹)", value=2000, min_value=100, step=100)

        years_back = st.slider("Years of Data", 1, 5, 2)
        end_date = date.today()
        start_date = end_date - timedelta(days=365 * years_back)

        st.markdown("---")
        run_btn = st.button("Run Backtest", type="primary")

    col_left, col_right = st.columns([2, 1])

    if run_btn:
        with st.spinner("Loading data & running engine..."):
            df = load_data(ticker, start_date, end_date, interval)

            df = df[~df.index.duplicated(keep='first')].copy()
            df = df.sort_index()

            if df.empty:
                st.error("No data loaded. Check ticker or timeframe.")
                return

            df, div_pairs = apply_divergence_engine(df)

            trades, final_equity = run_backtest(df, div_pairs, risk_per_trade=risk_per_trade)
            trades_df = trades_to_df(trades)

            total_trades, win_rate, net_pnl, max_dd = compute_metrics(trades_df)

            price_fig, rsi_fig, atr_fig = plot_chart(df, trades_df)
            equity_curve = build_equity_curve(trades_df)

        with col_left:
            st.subheader(f"{ticker} – Price & Signals")
            st.plotly_chart(price_fig, use_container_width=True)

            st.subheader("RSI & Divergence")
            st.plotly_chart(rsi_fig, use_container_width=True)

            st.subheader("ATR")
            st.plotly_chart(atr_fig, use_container_width=True)

            if len(equity_curve) > 0:
                eq_fig = go.Figure()
                eq_fig.add_trace(go.Scatter(
                    x=trades_df['Exit'],
                    y=equity_curve,
                    mode='lines+markers',
                    name='Equity',
                    line=dict(color='cyan', width=2)
                ))
                eq_fig.update_layout(
                    height=250,
                    margin=dict(l=10, r=10, t=30, b=10),
                    title="Equity Curve"
                )
                st.subheader("Equity Curve")
                st.plotly_chart(eq_fig, use_container_width=True)
            else:
                st.info("No trades → no equity curve.")

        with col_right:
            st.subheader("Backtest Results")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Total Trades", total_trades)
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with c2:
                st.metric("Net PnL (₹)", f"{net_pnl:,.0f}")
                st.metric("Max Drawdown (₹)", f"{max_dd:,.0f}")

            st.markdown("---")
            st.subheader("Trade Details")

            if len(trades_df) > 0:
                st.dataframe(
                    trades_df.style.format({
                        "EntryPrice": "{:.2f}",
                        "ExitPrice": "{:.2f}",
                        "Qty": "{:.0f}",
                        "PnL": "{:.2f}",
                        "Equity": "{:.2f}",
                    }),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No trades generated by the divergence engine in this period.")
    else:
        with col_left:
            st.info("Set your parameters in the sidebar and click **Run Backtest**.")
        with col_right:
            st.empty()

if __name__ == "__main__":
    main()
