import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta
import pandas as pd

from utils.indicators import ema, rsi, atr, avwap
from utils.divergence import apply_divergence_engine
from utils.backtester import run_backtest, trades_to_df, build_equity_curve


# ============================
# DATA LOADER
# ============================
@st.cache_data
def load_data(ticker, start, end, interval):
    df = yf.download(ticker, start=start, end=end, interval=interval)
    df.dropna(inplace=True)
    return df


# ============================
# CHART BUILDER
# ============================
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

    # Trade markers
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


# ============================
# METRICS
# ============================
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


# ============================
# STREAMLIT APP
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

        years_back = st.slider("Years of Data", 1, 5, 2)
        end_date = date.today()
        start_date = end_date - timedelta(days=365 * years_back)

        st.markdown("---")
        run_btn = st.button("Run Backtest", type="primary")

    col_left, col_right = st.columns([2, 1])

    if run_btn:
        with st.spinner("Loading data & running engine..."):
            df = load_data(ticker, start_date, end_date, interval)

            # ⭐ FIX: remove duplicate timestamps and sort
            df = df[~df.index.duplicated(keep='first')].copy()
            df = df.sort_index()

            if df.empty:
                st.error("No data loaded. Check ticker or timeframe.")
                return

            # Divergence engine
            df, div_pairs = apply_divergence_engine(df)

            # Backtest
            trades, final_equity = run_backtest(df, div_pairs, risk_per_trade=risk_per_trade)
            trades_df = trades_to_df(trades)

            # Metrics
            total_trades, win_rate, net_pnl, max_dd = compute_metrics(trades_df)

            # Charts
            price_fig, rsi_fig, atr_fig = plot_chart(df, trades_df)
            equity_curve = build_equity_curve(trades_df)

        # LEFT SIDE
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

        # RIGHT SIDE
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
