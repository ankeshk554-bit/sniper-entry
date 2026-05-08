import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sniper Terminal · Ankesh",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Terminal Aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

:root {
    --bg0:      #08090d;
    --bg1:      #0e1018;
    --bg2:      #13161f;
    --bg3:      #1a1e2a;
    --border:   #1f2436;
    --border2:  #2a3050;
    --text:     #c8cfdf;
    --muted:    #505872;
    --green:    #00e676;
    --red:      #ff3d5a;
    --gold:     #ffc107;
    --cyan:     #00b8d9;
    --accent:   #4f6ef7;
}

html, body, [class*="css"] {
    background-color: var(--bg0) !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 1.8rem !important; max-width: 100% !important; }

[data-testid="stSidebar"] {
    background: var(--bg1) !important;
    border-right: 1px solid var(--border2) !important;
    width: 280px !important;
}

[data-testid="stMetric"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 1rem 1.2rem !important;
}

.watchlist-chip {
    background: var(--bg3);
    border: 1px solid var(--border2);
    border-radius: 4px;
    padding: 0.4rem 0.75rem;
    font-size: 0.72rem;
    color: var(--cyan);
    margin: 2px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "TITAN.NS", "INFY.NS", "HDFCBANK.NS"]

with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.2rem'>
        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:white;letter-spacing:0.04em'>
            ⚡ SNIPER <span style='color:#4f6ef7'>TERMINAL</span>
        </div>
        <div style='font-size:0.6rem;color:#505872;letter-spacing:0.12em;margin-top:2px'>SWING ENGINE v2.2</div>
    </div>
    """, unsafe_allow_html=True)

    ticker = st.text_input("Search Ticker", value="RELIANCE.NS").upper()
    
    tf_options = {"1W — Weekly": "1wk", "1D — Daily": "1d", "4H — 4 Hour": "4h"}
    selected_tf_label = st.selectbox("Timeframe", list(tf_options.keys()), index=1)
    interval = tf_options[selected_tf_label]

    risk_amt = st.number_input("Risk per Trade (₹)", value=2000, step=500)

    use_mtf = st.checkbox("Use Weekly Trend Filter (MTF)", value=True)

    st.markdown("### Watchlist")
    chips_html = "".join([f"<span class='watchlist-chip'>{t}</span>" for t in st.session_state.watchlist])
    st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INDICATOR HELPERS (FAST)
# ─────────────────────────────────────────────
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fast_avwap(df, start_idx):
    c = df['Close'].values[start_idx:]
    v = df['Volume'].values[start_idx:]
    cv = np.cumsum(c * v)
    vv = np.cumsum(v)
    out = np.full(len(df), np.nan)
    out[start_idx:] = cv / vv
    return out

# ─────────────────────────────────────────────
# BACKTEST ENGINE (WITH WEEKLY TREND FILTER)
# ─────────────────────────────────────────────
def run_backtest(df, risk_per_trade=2000, use_mtf=True):
    df = df.copy()

    # Daily trend
    df['Trend'] = df['Close'] > df['EMA200']

    # Entry condition (MTF optional)
    if use_mtf:
        df['Entry'] = (
            (df['RSI'] < 32) &
            (df['Close'] > df['AVWAP_BOT']) &
            (df['Trend']) &
            (df['TrendW'])     # Weekly trend
        )
    else:
        df['Entry'] = (
            (df['RSI'] < 32) &
            (df['Close'] > df['AVWAP_BOT']) &
            (df['Trend'])
        )

    # Stoploss = previous swing low
    df['SL'] = df['Low'].rolling(5).min().shift(1)

    # Clean SL
    df['SL'] = df['SL'].replace([np.inf, -np.inf], np.nan)
    df['SL'] = df['SL'].bfill()
    df['SL'] = df['SL'].fillna(df['Low'])

    # Risk
    df['Risk'] = df['Close'] - df['SL']
    df['Risk'] = df['Risk'].replace([np.inf, -np.inf], np.nan)
    df['Risk'] = df['Risk'].fillna(0)

    # Position size
    df['Qty'] = risk_per_trade / df['Risk']
    df['Qty'] = df['Qty'].replace([np.inf, -np.inf], np.nan)
    df['Qty'] = df['Qty'].fillna(0)
    df['Qty'] = df['Qty'].clip(lower=0).astype(int)

    trades = []
    in_trade = False

    for i in range(len(df)):
        if (not in_trade) and df['Entry'].iloc[i] and df['Qty'].iloc[i] > 0 and df['Risk'].iloc[i] > 0:
            entry_i = i
            entry_price = float(df['Close'].iloc[i])
            sl = float(df['SL'].iloc[i])
            risk_per_share = entry_price - sl
            tgt = entry_price + 2 * risk_per_share
            qty = int(df['Qty'].iloc[i])
            in_trade = True
            continue

        if in_trade:
            low = float(df['Low'].iloc[i])
            high = float(df['High'].iloc[i])

            if low <= sl:
                trades.append({
                    'Entry': df.index[entry_i],
                    'Exit': df.index[i],
                    'EntryPrice': entry_price,
                    'ExitPrice': sl,
                    'Qty': qty,
                    'PnL': (sl - entry_price) * qty
                })
                in_trade = False
                continue

            if high >= tgt:
                trades.append({
                    'Entry': df.index[entry_i],
                    'Exit': df.index[i],
                    'EntryPrice': entry_price,
                    'ExitPrice': tgt,
                    'Qty': qty,
                    'PnL': (tgt - entry_price) * qty
                })
                in_trade = False
                continue

    if len(trades) == 0:
        return pd.DataFrame(columns=['Entry', 'Exit', 'EntryPrice', 'ExitPrice', 'Qty', 'PnL'])

    return pd.DataFrame(trades)

# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────
st.title(f"⚡ Terminal Scan: {ticker}")

try:
    period = "2y" if interval in ["1d", "1wk"] else "60d"
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)

    if not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Daily indicators
        df['EMA200'] = ema(df['Close'], 200)
        df['EMA50']  = ema(df['Close'], 50)
        df['RSI']    = rsi(df['Close'], 14)

        # Weekly trend filter
        df_w = yf.download(ticker, period="5y", interval="1wk", auto_adjust=True)
        df_w['EMA200'] = ema(df_w['Close'], 200)
        df_w['TrendW'] = df_w['Close'] > df_w['EMA200']

        df['TrendW'] = df_w['TrendW'].reindex(df.index, method='ffill')

        # AVWAP
        t_idx = int(df['High'].argmax())
        b_idx = int(df['Low'].argmin())
        df['AVWAP_TOP'] = fast_avwap(df, t_idx)
        df['AVWAP_BOT'] = fast_avwap(df, b_idx)

        # ─────────────────────────────────────────
        # PRICE + VOLUME + RSI CHART
        # ─────────────────────────────────────────
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.15, 0.25]
        )

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA200'],
            line=dict(color="#4f6ef7", width=1.5, dash="dot"),
            name="EMA 200"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df['AVWAP_TOP'],
            line=dict(color="#ff3d5a", width=1.5),
            name="AVWAP Supply"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df['AVWAP_BOT'],
            line=dict(color="#00e676", width=1.5),
            name="AVWAP Support"
        ), row=1, col=1)

        # Volume
        v_colors = ["#00e676" if df['Close'].iloc[i] > df['Open'].iloc[i] else "#ff3d5a" for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'],
            marker_color=v_colors,
            name="Volume"
        ), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'],
            line=dict(color="#00b8d9", width=1.5),
            name="RSI"
        ), row=3, col=1)

        fig.add_hline(y=70, line_dash="dash", line_color="#ff3d5a", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00e676", row=3, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=800,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="#08090d",
            plot_bgcolor="#08090d",
            margin=dict(l=0, r=0, b=0, t=10)
        )

        st.plotly_chart(fig, use_container_width=True)

        # ─────────────────────────────────────────
        # BACKTEST RESULTS
        # ─────────────────────────────────────────
        st.subheader("📈 Backtest Results")

        bt = run_backtest(df, risk_amt, use_mtf)

        if bt.empty:
            st.warning("No trades triggered for this strategy.")
        else:
            bt['Equity'] = bt['PnL'].cumsum()
            max_dd = (bt['Equity'].cummax() - bt['Equity']).max()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Trades", len(bt))
            c2.metric("Win Rate", f"{(bt['PnL'] > 0).mean()*100:.1f}%")
            c3.metric("Net PnL", f"₹{bt['PnL'].sum():,.0f}")
            c4.metric("Max Drawdown", f"₹{max_dd:,.0f}")

            st.dataframe(bt)

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=bt['Exit'],
                y=bt['Equity'],
                mode='lines',
                line=dict(color="#4f6ef7")
            ))
            fig_bt.update_layout(
                height=300,
                template="plotly_dark",
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="#08090d",
                plot_bgcolor="#08090d",
            )
            st.plotly_chart(fig_bt, use_container_width=True)

        # ─────────────────────────────────────────
        # EXECUTION PLAN
        # ─────────────────────────────────────────
        curr = df.iloc[-1]
        entry = float(curr['Close'])
        sl_live = float(curr['Low'])
        diff = entry - sl_live
        qty_live = int(risk_amt / diff) if diff > 0 else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Live Price", f"₹{entry:,.2f}")
        c2.metric(f"Qty (Risk ₹{risk_amt})", f"{qty_live}")
        c3.metric("Stop Loss", f"₹{sl_live:,.2f}", delta=f"-₹{diff:.2f}", delta_color="inverse")
        c4.metric("Target (1:2)", f"₹{entry + (diff*2):,.2f}" if diff > 0 else "—")

    else:
        st.error("No data returned for this ticker/timeframe.")

except Exception as e:
    st.error(f"Analysis Error: {e}")
