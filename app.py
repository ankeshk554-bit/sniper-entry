import numpy as np
import pandas as pd

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
    """
    Strict swing low:
    Low < Low[-1], Low < Low[-2]
    Low < Low[+1], Low < Low[+2]
    """
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
    """
    Classic bullish divergence:
    Price: Lower Low
    RSI: Higher Low
    Both must be swing lows.
    """
    divergence_points = []
    lows = df['Low'].values
    rsi_vals = df['RSI'].values

    swing_indices = np.where(swing_low_mask)[0]

    for i in range(1, len(swing_indices)):
        i1 = swing_indices[i - 1]
        i2 = swing_indices[i]

        # Price LL + RSI HL
        if lows[i2] < lows[i1] and rsi_vals[i2] > rsi_vals[i1]:
            divergence_points.append((i1, i2))

    return divergence_points

# ============================
# 7. DIVERGENCE MARKER GENERATOR
# ============================
def generate_divergence_markers(df, divergence_pairs):
    """
    Creates marker arrays for plotting:
    - Price arrows
    - Price connecting lines
    - RSI connecting lines
    """
    df['Div_Arrow'] = np.nan
    df['Div_Line_Price'] = np.nan
    df['Div_Line_RSI'] = np.nan

    for (i1, i2) in divergence_pairs:
        # Arrow at second swing low
        df.loc[i2, 'Div_Arrow'] = df['Low'].iloc[i2] * 0.995

        # Price line
        df.loc[i1:i2, 'Div_Line_Price'] = np.linspace(
            df['Low'].iloc[i1],
            df['Low'].iloc[i2],
            i2 - i1 + 1
        )

        # RSI line
        df.loc[i1:i2, 'Div_Line_RSI'] = np.linspace(
            df['RSI'].iloc[i1],
            df['RSI'].iloc[i2],
            i2 - i1 + 1
        )

    return df

# ============================
# 8. FULL DIVERGENCE ENGINE
# ============================
def apply_divergence_engine(df):
    # Indicators
    df['EMA200'] = ema(df['Close'], 200)
    df['RSI'] = rsi(df['Close'])
    df['ATR'] = atr(df)
    df['AVWAP'] = avwap(df)

    # Strict swing lows
    swing_mask = detect_strict_swing_lows(df)
    df['SwingLow'] = swing_mask

    # Divergence pairs
    div_pairs = detect_rsi_bullish_divergence(df, swing_mask)

    # Markers
    df = generate_divergence_markers(df, div_pairs)

    return df, div_pairs
# ============================================
# BLOCK 2 — BACKTEST ENGINE
# ============================================

def run_backtest(df, divergence_pairs, risk_per_trade=2000):
    """
    Backtest logic:
    - Entry only on valid divergence (strict)
    - Trend filter: Close > EMA200
    - AVWAP filter: Close > AVWAP
    - SL = 1.5 × ATR
    - TP = 2 × ATR
    - Candle-based exits
    """

    trades = []
    equity = 0
    df = df.copy()

    # Convert divergence pairs into a set of entry indices
    divergence_entries = {i2 for (_, i2) in divergence_pairs}

    for i in range(len(df)):
        # Only enter on divergence
        if i not in divergence_entries:
            continue

        # Trend filter
        if df['Close'].iloc[i] < df['EMA200'].iloc[i]:
            continue

        # AVWAP filter
        if df['Close'].iloc[i] < df['AVWAP'].iloc[i]:
            continue

        entry_price = df['Close'].iloc[i]
        atr_val = df['ATR'].iloc[i]

        if np.isnan(atr_val) or atr_val == 0:
            continue

        # Position sizing
        sl_distance = 1.5 * atr_val
        qty = max(int(risk_per_trade / sl_distance), 1)

        sl_price = entry_price - sl_distance
        tp_price = entry_price + (2 * atr_val)

        # Simulate forward candles
        exit_price = None
        exit_index = None

        for j in range(i + 1, len(df)):
            low_j = df['Low'].iloc[j]
            high_j = df['High'].iloc[j]

            # SL hit
            if low_j <= sl_price:
                exit_price = sl_price
                exit_index = j
                break

            # TP hit
            if high_j >= tp_price:
                exit_price = tp_price
                exit_index = j
                break

        # If no exit found, exit at last close
        if exit_price is None:
            exit_price = df['Close'].iloc[-1]
            exit_index = len(df) - 1

        pnl = (exit_price - entry_price) * qty
        equity += pnl

        trades.append({
            "Entry": df.index[i],
            "Exit": df.index[exit_index],
            "EntryPrice": entry_price,
            "ExitPrice": exit_price,
            "Qty": qty,
            "PnL": pnl,
            "Equity": equity
        })

    return trades, equity


# ============================================
# TRADE LIST → DATAFRAME
# ============================================
def trades_to_df(trades):
    if len(trades) == 0:
        return pd.DataFrame(columns=[
            "Entry", "Exit", "EntryPrice", "ExitPrice", "Qty", "PnL", "Equity"
        ])
    return pd.DataFrame(trades)


# ============================================
# EQUITY CURVE GENERATOR
# ============================================
def build_equity_curve(trades_df):
    if len(trades_df) == 0:
        return pd.Series(dtype=float)
    return trades_df["Equity"]
# ============================================
# BLOCK 3 — STREAMLIT APP (UI + CHART + BACKTEST)
# ============================================
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

# ====== IMPORT / MERGE BLOCK 1 + BLOCK 2 HERE IF IN SEPARATE FILES ======
# From previous blocks:
# - ema, rsi, atr, avwap
# - detect_strict_swing_lows, detect_rsi_bullish_divergence
# - generate_divergence_markers, apply_divergence_engine
# - run_backtest, trades_to_df, build_equity_curve
# (If all in one file, they are already defined above.)

# ============================================
# DATA LOADER
# ============================================
@st.cache_data
def load_data(ticker, start, end, interval):
    df = yf.download(ticker, start=start, end=end, interval=interval)
    df.dropna(inplace=True)
    return df

# ============================================
# CHART BUILDER
# ============================================
def plot_chart(df, trades_df):
    fig = go.Figure()

    # Price candles
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price"
    ))

    # EMA200
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA200'],
        mode='lines',
        name='EMA200',
        line=dict(color='orange', width=1)
    ))

    # AVWAP
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['AVWAP'],
        mode='lines',
        name='AVWAP',
        line=dict(color='purple', width=1)
    ))

    # Divergence price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Div_Line_Price'],
        mode='lines',
        name='Div Price Line',
        line=dict(color='lime', width=1, dash='dot'),
        connectgaps=False
    ))

    # Divergence arrows
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Div_Arrow'],
        mode='markers',
        name='Bullish Div',
        marker=dict(symbol='triangle-up', size=10, color='lime'),
    ))

    # Mark trade entries/exits
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

    # RSI panel
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

    # ATR panel
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

# ============================================
# METRICS CALC
# ============================================
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
        max_dd_val = 0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd_val:
                max_dd_val = dd
        max_dd = max_dd_val

    return total_trades, win_rate, net_pnl, max_dd

# ============================================
# STREAMLIT APP
# ============================================
def main():
    st.set_page_config(page_title="Sniper Terminal – Ankesh", layout="wide")

    st.title("Sniper Terminal – Ankesh")
    st.caption("Strict RSI Bullish Divergence • Trend + AVWAP + ATR Engine")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        ticker = st.text_input("Ticker", value="HAL.NS")
        interval = st.selectbox("Timeframe", ["1d", "1h", "15m"], index=0)
        risk_per_trade = st.number_input("Risk per Trade (₹)", value=2000, min_value=100, step=100)

        years_back = st.slider("Years of Data", 1, 5, 2)
        end_date = date.today()
        start_date = end_date - timedelta(days=365 * years_back)

        st.markdown("---")
        st.write("Click **Run Backtest** to execute the divergence engine.")

        run_btn = st.button("Run Backtest", type="primary")

    # Layout
    col_left, col_right = st.columns([2, 1])

    if run_btn:
        with st.spinner("Loading data & running engine..."):
            df = load_data(ticker, start_date, end_date, interval)

            if df.empty:
                st.error("No data loaded. Check ticker or timeframe.")
                return

            # Apply divergence engine
            df, div_pairs = apply_divergence_engine(df)

            # Run backtest
            trades, final_equity = run_backtest(df, div_pairs, risk_per_trade=risk_per_trade)
            trades_df = trades_to_df(trades)

            # Metrics
            total_trades, win_rate, net_pnl, max_dd = compute_metrics(trades_df)

            # Charts
            price_fig, rsi_fig, atr_fig = plot_chart(df, trades_df)
            equity_curve = build_equity_curve(trades_df)

        # LEFT: Charts
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

        # RIGHT: Stats + Trades
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
