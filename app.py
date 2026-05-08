import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ─────────────────────────────────────────────

# PAGE CONFIG

# ─────────────────────────────────────────────

st.set_page_config(
page_title=“Sniper Terminal · Ankesh”,
layout=“wide”,
initial_sidebar_state=“expanded”,
)

# ─────────────────────────────────────────────

# GLOBAL CSS  —  Terminal Aesthetic

# ─────────────────────────────────────────────

st.markdown(”””

<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

/* ── Root variables ── */
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

/* ── Base ── */
html, body, [class*="css"] {
    background-color: var(--bg0) !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 1.8rem !important; max-width: 100% !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg1) !important;
    border-right: 1px solid var(--border2) !important;
    width: 280px !important;
}
[data-testid="stSidebar"] .css-1d391kg { padding: 1.2rem !important; }
[data-testid="stSidebar"] label {
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

/* ── Input fields ── */
input, .stTextInput > div > div > input {
    background: var(--bg2) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(79,110,247,0.15) !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--bg2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 4px !important;
    font-size: 0.82rem !important;
}

/* ── Number input ── */
.stNumberInput > div > div > input {
    background: var(--bg2) !important;
    border: 1px solid var(--border2) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 1rem 1.2rem !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--cyan));
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.6rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Sidebar ticker chip ── */
.watchlist-chip {
    background: var(--bg3);
    border: 1px solid var(--border2);
    border-radius: 4px;
    padding: 0.4rem 0.75rem;
    font-size: 0.72rem;
    color: var(--cyan);
    cursor: pointer;
    display: inline-block;
    margin: 2px;
    letter-spacing: 0.06em;
    transition: all 0.15s;
}
.watchlist-chip:hover { background: var(--accent); color: white; border-color: var(--accent); }

/* ── Section headers ── */
.section-label {
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Status badge ── */
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green);
    display: inline-block;
    box-shadow: 0 0 6px var(--green);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Top bar ── */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}
.terminal-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    color: white;
}
.terminal-title span { color: var(--accent); }
.ts-badge {
    font-size: 0.6rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Execution plan card ── */
.exec-card {
    background: var(--bg2);
    border: 1px solid var(--border2);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
.exec-title {
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}
.risk-bar {
    height: 3px;
    background: linear-gradient(90deg, var(--green) 0%, var(--gold) 50%, var(--red) 100%);
    border-radius: 2px;
    margin-top: 1rem;
    opacity: 0.7;
}

/* ── Error ── */
.stAlert { border-radius: 6px !important; font-size: 0.8rem !important; }

/* ── Chart container ── */
.js-plotly-plot { border-radius: 6px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg1); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }
</style>

“””, unsafe_allow_html=True)

# ─────────────────────────────────────────────

# SESSION STATE

# ─────────────────────────────────────────────

if ‘watchlist’ not in st.session_state:
st.session_state.watchlist = [“RELIANCE.NS”, “TCS.NS”, “TITAN.NS”, “INFY.NS”, “HDFCBANK.NS”]

# ─────────────────────────────────────────────

# SIDEBAR

# ─────────────────────────────────────────────

with st.sidebar:
st.markdown(”””
<div style='padding:0.5rem 0 1.2rem'>
<div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:white;letter-spacing:0.04em'>
⚡ SNIPER <span style='color:#4f6ef7'>TERMINAL</span>
</div>
<div style='font-size:0.6rem;color:#505872;letter-spacing:0.12em;margin-top:2px'>SWING TRADING ENGINE v2.1</div>
</div>
“””, unsafe_allow_html=True)

```
# Live status
now = datetime.now().strftime("%H:%M:%S")
st.markdown(f"""
<div style='display:flex;align-items:center;gap:8px;background:#0e1018;border:1px solid #1f2436;
            border-radius:4px;padding:0.5rem 0.75rem;margin-bottom:1rem;font-size:0.68rem;color:#505872'>
    <span class='status-dot'></span>
    FEED ACTIVE &nbsp;·&nbsp; {now}
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='section-label'>Search Ticker</div>", unsafe_allow_html=True)
ticker = st.text_input("", value="RELIANCE.NS", label_visibility="collapsed").upper()

st.markdown("<div class='section-label' style='margin-top:1rem'>Timeframe</div>", unsafe_allow_html=True)
tf_options = {
    "1W — Weekly":   "1wk",
    "1D — Daily":    "1d",
    "4H — 4 Hour":   "4h",
    "1H — 1 Hour":   "1h",
    "15M — 15 Min":  "15m",
}
selected_tf_label = st.selectbox("", list(tf_options.keys()), index=1, label_visibility="collapsed")
interval = tf_options[selected_tf_label]
selected_tf = selected_tf_label.split("—")[0].strip()

st.markdown("<div class='section-label' style='margin-top:1rem'>Risk Management</div>", unsafe_allow_html=True)
risk_amt = st.number_input("Risk per Trade (₹)", value=2000, step=500, min_value=500, label_visibility="visible")

st.markdown("<div class='section-label' style='margin-top:1rem'>Watchlist</div>", unsafe_allow_html=True)
chips_html = "".join([f"<span class='watchlist-chip'>{t}</span>" for t in st.session_state.watchlist])
st.markdown(f"<div style='line-height:2'>{chips_html}</div>", unsafe_allow_html=True)

new_ticker = st.text_input("Add to watchlist", placeholder="e.g. WIPRO.NS", key="wl_input")
if st.button("＋ Add", use_container_width=True):
    if new_ticker and new_ticker.upper() not in st.session_state.watchlist:
        st.session_state.watchlist.append(new_ticker.upper())
        st.rerun()

st.markdown("""
<div style='position:absolute;bottom:1.5rem;left:1rem;right:1rem;
            font-size:0.55rem;color:#303650;text-align:center;letter-spacing:0.08em;line-height:1.8'>
    NOT FINANCIAL ADVICE<br>FOR EDUCATIONAL USE ONLY
</div>
""", unsafe_allow_html=True)
```

# ─────────────────────────────────────────────

# DIVERGENCE CALCULATOR

# ─────────────────────────────────────────────

def calculate_divergence_lines(df, window=5):
df[‘Price_Low’]  = df[‘Low’].rolling(window=window*2+1, center=True).min()
df[‘Price_High’] = df[‘High’].rolling(window=window*2+1, center=True).max()

```
bull_lines, bear_lines = [], []
last_p_low, last_p_high = -1, -1

for i in range(window*2, len(df)):
    if df['Low'].iloc[i] == df['Price_Low'].iloc[i]:
        if last_p_low != -1:
            if df['Low'].iloc[i] < df['Low'].iloc[last_p_low] and df['RSI'].iloc[i] > df['RSI'].iloc[last_p_low]:
                if df['RSI'].iloc[i] < 35:
                    bull_lines.append(((df.index[last_p_low], df['RSI'].iloc[last_p_low]),
                                       (df.index[i], df['RSI'].iloc[i])))
                    bull_lines.append(((df.index[last_p_low], df['Low'].iloc[last_p_low]),
                                       (df.index[i], df['Low'].iloc[i]), 'price'))
        last_p_low = i

    if df['High'].iloc[i] == df['Price_High'].iloc[i]:
        if last_p_high != -1:
            if df['High'].iloc[i] > df['High'].iloc[last_p_high] and df['RSI'].iloc[i] < df['RSI'].iloc[last_p_high]:
                if df['RSI'].iloc[i] > 65:
                    bear_lines.append(((df.index[last_p_high], df['RSI'].iloc[last_p_high]),
                                       (df.index[i], df['RSI'].iloc[i])))
                    bear_lines.append(((df.index[last_p_high], df['High'].iloc[last_p_high]),
                                       (df.index[i], df['High'].iloc[i]), 'price'))
        last_p_high = i

return bull_lines, bear_lines
```

# ─────────────────────────────────────────────

# MAIN DASHBOARD

# ─────────────────────────────────────────────

# Top bar

st.markdown(f”””

<div class='top-bar'>
    <div>
        <span class='terminal-title'>⚡ <span>{ticker}</span></span>
        <span style='font-size:0.65rem;color:#505872;margin-left:0.75rem;
                     letter-spacing:0.08em'>/ INSTITUTIONAL STRUCTURE SCANNER</span>
    </div>
    <div class='ts-badge'>{selected_tf} · {datetime.now().strftime("%d %b %Y  %H:%M")}</div>
</div>
""", unsafe_allow_html=True)

if ticker:
try:
data_period = “2y” if interval in [“1d”, “1wk”] else “60d”

```
    with st.spinner(f"Fetching {ticker} data..."):
        raw_df = yf.download(ticker, period=data_period, interval=interval,
                             progress=False, auto_adjust=True)

    if not raw_df.empty:
        df = raw_df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # ── Indicators ──
        df['EMA200']  = ta.ema(df['Close'], length=200)
        df['EMA50']   = ta.ema(df['Close'], length=50)
        df['RSI']     = ta.rsi(df['Close'], length=14)
        bull_lines, bear_lines = calculate_divergence_lines(df)

        # ── Anchored VWAP ──
        top_idx = df['High'].argmax()
        bot_idx = df['Low'].argmin()
        def calc_avwap(df, idx):
            temp = df.iloc[idx:].copy()
            return (temp['Close'] * temp['Volume']).cumsum() / temp['Volume'].cumsum()
        df['AVWAP_TOP'] = calc_avwap(df, top_idx)
        df['AVWAP_BOT'] = calc_avwap(df, bot_idx)

        # ── Quick stats for header ──
        curr      = df.iloc[-1]
        prev      = df.iloc[-2]
        chg       = float(curr['Close']) - float(prev['Close'])
        chg_pct   = (chg / float(prev['Close'])) * 100
        chg_color = "#00e676" if chg >= 0 else "#ff3d5a"
        chg_arrow = "▲" if chg >= 0 else "▼"
        rsi_val   = round(float(df['RSI'].iloc[-1]), 1)
        rsi_color = "#ff3d5a" if rsi_val > 70 else ("#00e676" if rsi_val < 30 else "#ffc107")

        # ── Price summary strip ──
        st.markdown(f"""
        <div style='display:flex;gap:1.5rem;flex-wrap:wrap;
                    background:#0e1018;border:1px solid #1a1e2a;border-radius:6px;
                    padding:0.85rem 1.2rem;margin-bottom:1rem;align-items:center'>
            <div>
                <div style='font-size:0.55rem;letter-spacing:0.14em;color:#505872;
                            text-transform:uppercase;margin-bottom:2px'>Last Price</div>
                <div style='font-size:1.6rem;font-weight:600;color:white;letter-spacing:-0.02em'>
                    ₹{round(float(curr['Close']),2):,}
                </div>
            </div>
            <div style='border-left:1px solid #1f2436;padding-left:1.5rem'>
                <div style='font-size:0.55rem;letter-spacing:0.14em;color:#505872;
                            text-transform:uppercase;margin-bottom:2px'>Change</div>
                <div style='font-size:1.1rem;font-weight:600;color:{chg_color}'>
                    {chg_arrow} ₹{abs(round(chg,2))} ({round(chg_pct,2):+.2f}%)
                </div>
            </div>
            <div style='border-left:1px solid #1f2436;padding-left:1.5rem'>
                <div style='font-size:0.55rem;letter-spacing:0.14em;color:#505872;
                            text-transform:uppercase;margin-bottom:2px'>RSI (14)</div>
                <div style='font-size:1.1rem;font-weight:600;color:{rsi_color}'>{rsi_val}</div>
            </div>
            <div style='border-left:1px solid #1f2436;padding-left:1.5rem'>
                <div style='font-size:0.55rem;letter-spacing:0.14em;color:#505872;
                            text-transform:uppercase;margin-bottom:2px'>Day High</div>
                <div style='font-size:1.1rem;color:#c8cfdf'>₹{round(float(curr['High']),2):,}</div>
            </div>
            <div style='border-left:1px solid #1f2436;padding-left:1.5rem'>
                <div style='font-size:0.55rem;letter-spacing:0.14em;color:#505872;
                            text-transform:uppercase;margin-bottom:2px'>Day Low</div>
                <div style='font-size:1.1rem;color:#c8cfdf'>₹{round(float(curr['Low']),2):,}</div>
            </div>
            <div style='border-left:1px solid #1f2436;padding-left:1.5rem'>
                <div style='font-size:0.55rem;letter-spacing:0.14em;color:#505872;
                            text-transform:uppercase;margin-bottom:2px'>Volume</div>
                <div style='font-size:1.1rem;color:#c8cfdf'>{int(curr["Volume"]):,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ──────────────────────────────
        #  PLOTLY CHART
        # ──────────────────────────────
        CHART_BG  = "#08090d"
        GRID_COL  = "#13161f"
        TEXT_COL  = "#505872"

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.58, 0.18, 0.24],
        )

        # ── Candlesticks ──
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'],   close=df['Close'],
            increasing=dict(line=dict(color="#00e676", width=1), fillcolor="#00e676"),
            decreasing=dict(line=dict(color="#ff3d5a", width=1), fillcolor="#ff3d5a"),
            name="Price",
        ), row=1, col=1)

        # ── EMA overlays ──
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA200'],
            line=dict(color="#4f6ef7", width=1.5, dash="dot"),
            name="EMA 200", opacity=0.85,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['EMA50'],
            line=dict(color="#ffc107", width=1.2, dash="dot"),
            name="EMA 50", opacity=0.75,
        ), row=1, col=1)

        # ── VWAP overlays ──
        fig.add_trace(go.Scatter(
            x=df.index, y=df['AVWAP_TOP'],
            line=dict(color="#ff3d5a", width=1.5),
            name="AVWAP Supply", opacity=0.9,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['AVWAP_BOT'],
            line=dict(color="#00e676", width=1.5),
            name="AVWAP Support", opacity=0.9,
        ), row=1, col=1)

        # ── Divergence price lines ──
        for line in bull_lines:
            if len(line) == 3 and line[2] == 'price':
                fig.add_trace(go.Scatter(
                    x=[line[0][0], line[1][0]], y=[line[0][1], line[1][1]],
                    mode='lines+markers',
                    line=dict(color='#00e676', width=2, dash='dot'),
                    marker=dict(size=6, symbol='circle', color='#00e676'),
                    showlegend=False,
                ), row=1, col=1)
        for line in bear_lines:
            if len(line) == 3 and line[2] == 'price':
                fig.add_trace(go.Scatter(
                    x=[line[0][0], line[1][0]], y=[line[0][1], line[1][1]],
                    mode='lines+markers',
                    line=dict(color='#ff3d5a', width=2, dash='dot'),
                    marker=dict(size=6, symbol='circle', color='#ff3d5a'),
                    showlegend=False,
                ), row=1, col=1)

        # ── Volume bars ──
        vol_colors = [
            "#00c853" if float(df['Open'].iloc[i]) < float(df['Close'].iloc[i]) else "#d50000"
            for i in range(len(df))
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'],
            marker_color=vol_colors,
            marker_opacity=0.7,
            name="Volume",
        ), row=2, col=1)

        # ── RSI ──
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'],
            line=dict(color="#00b8d9", width=1.5),
            name="RSI",
            fill='tozeroy',
            fillcolor="rgba(0,184,217,0.04)",
        ), row=3, col=1)

        fig.add_hline(y=70, line=dict(color="#ff3d5a", width=1, dash="dash"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="#00e676", width=1, dash="dash"), row=3, col=1)
        fig.add_hline(y=50, line=dict(color="#303650",  width=1, dash="dot"),  row=3, col=1)

        # ── RSI divergence lines ──
        for line in bull_lines:
            if len(line) == 2:
                fig.add_trace(go.Scatter(
                    x=[line[0][0], line[1][0]], y=[line[0][1], line[1][1]],
                    mode='lines+markers',
                    line=dict(color='#00e676', width=2),
                    marker=dict(size=5, symbol='circle', color='#00e676'),
                    name="Bull Div",
                ), row=3, col=1)
        for line in bear_lines:
            if len(line) == 2:
                fig.add_trace(go.Scatter(
                    x=[line[0][0], line[1][0]], y=[line[0][1], line[1][1]],
                    mode='lines+markers',
                    line=dict(color='#ff3d5a', width=2),
                    marker=dict(size=5, symbol='circle', color='#ff3d5a'),
                    name="Bear Div",
                ), row=3, col=1)

        # ── Layout ──
        fig.update_layout(
            template="plotly_dark",
            height=760,
            xaxis_rangeslider_visible=False,
            showlegend=False,
            plot_bgcolor=CHART_BG,
            paper_bgcolor=CHART_BG,
            margin=dict(l=0, r=0, b=0, t=12),
            font=dict(family="JetBrains Mono, monospace", size=11, color=TEXT_COL),
        )

        # Axis styling for all subplots
        for axis in ['xaxis', 'xaxis2', 'xaxis3', 'yaxis', 'yaxis2', 'yaxis3']:
            fig.update_layout(**{axis: dict(
                gridcolor=GRID_COL,
                linecolor="#1a1e2a",
                tickfont=dict(size=10, color=TEXT_COL),
                showgrid=True,
                zeroline=False,
            )})

        fig.update_layout(yaxis3=dict(range=[0, 100]))

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ──────────────────────────────
        #  EXECUTION PLAN
        # ──────────────────────────────
        entry = float(curr['Close'])
        sl    = float(curr['Low'])
        diff  = entry - sl
        qty   = int(risk_amt / diff) if diff > 0 else 0
        tgt1  = entry + diff
        tgt2  = entry + diff * 2
        tgt3  = entry + diff * 3

        st.markdown("<div class='section-label' style='margin-top:0.5rem'>Execution Plan</div>",
                    unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Entry Price",    f"₹{entry:,.2f}")
        c2.metric("Stop Loss",      f"₹{sl:,.2f}",    delta=f"-₹{diff:.2f}", delta_color="inverse")
        c3.metric("Qty  (R=₹2k)",   f"{qty} shares")
        c4.metric("Target 1:2",     f"₹{tgt2:,.2f}",  delta=f"+₹{diff*2:.2f}")
        c5.metric("Target 1:3",     f"₹{tgt3:,.2f}",  delta=f"+₹{diff*3:.2f}")

        # Risk/reward bar
        total_cap   = qty * entry
        max_loss    = qty * diff
        max_gain_2r = qty * diff * 2

        st.markdown(f"""
        <div class='exec-card'>
            <div class='exec-title'>Trade Summary · {ticker}</div>
            <div style='display:flex;gap:2.5rem;flex-wrap:wrap;font-size:0.75rem'>
                <div><span style='color:#505872'>Capital Required</span>&nbsp;&nbsp;
                     <span style='color:white'>₹{total_cap:,.0f}</span></div>
                <div><span style='color:#505872'>Max Risk</span>&nbsp;&nbsp;
                     <span style='color:#ff3d5a'>₹{max_loss:,.0f}</span></div>
                <div><span style='color:#505872'>Reward @ 1:2</span>&nbsp;&nbsp;
                     <span style='color:#00e676'>₹{max_gain_2r:,.0f}</span></div>
                <div><span style='color:#505872'>R:R Ratio</span>&nbsp;&nbsp;
                     <span style='color:#ffc107'>1 : 2.0</span></div>
                <div><span style='color:#505872'>Bull Divs Found</span>&nbsp;&nbsp;
                     <span style='color:#00e676'>{len([l for l in bull_lines if len(l)==2])}</span></div>
                <div><span style='color:#505872'>Bear Divs Found</span>&nbsp;&nbsp;
                     <span style='color:#ff3d5a'>{len([l for l in bear_lines if len(l)==2])}</span></div>
            </div>
            <div class='risk-bar'></div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error(f"No data returned for **{ticker}**. Check the ticker symbol.")

except Exception as e:
    st.error(f"Analysis Error: {e}")
```