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

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")

# Optional libs
try:
    import ta
except:
    ta = None

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    AGGRID_AVAILABLE = True
except:
    AGGRID_AVAILABLE = False

# ============================================================
# GLOBAL STATE INITIALIZATION
# ============================================================
if "settings" not in st.session_state:
    st.session_state["settings"] = {
        "gold_glow": 0.6,
        "default_tf": "1d",
        "default_risk": 2000,
        "default_years": 2,
        "show_rsi": True,
        "show_ema": True,
        "show_avwap": True,
        "show_vwap_top": True,
        "show_vwap_bottom": True,
        "show_divergences": True
    }

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = []

if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = False

if "last_scan" not in st.session_state:
    st.session_state["last_scan"] = None

if "layout_mode" not in st.session_state:
    st.session_state["layout_mode"] = "Analyst"

if "screener_results" not in st.session_state:
    st.session_state["screener_results"] = None

if "trade_history" not in st.session_state:
    st.session_state["trade_history"] = []

if "auto_trader" not in st.session_state:
    st.session_state["auto_trader"] = {
        "enabled": False,
        "positions": {},
        "trade_log": []
    }

# Dummy universe
NIFTY200 = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HAL.NS"]


# ============================================================
# NOTIFICATION SYSTEM
# ============================================================
def notify(message, type="info"):
    colors = {
        "info": "#D4AF37",
        "success": "#4caf50",
        "warning": "#ffc107",
        "error": "#f44336"
    }
    st.markdown(
        f"""
        <div style="
            padding:10px;
            margin-bottom:10px;
            border-radius:8px;
            background:rgba(20,20,20,0.85);
            border-left:4px solid {colors.get(type, '#D4AF37')};
            color:white;
            box-shadow:0px 0px 10px rgba(212,175,55,0.4);
        ">
            {message}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATA + INDICATORS
# ============================================================
@st.cache_data
def load_data(ticker, interval="1d", years=2):
    try:
        period = f"{years}y"
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        return df
    except:
        return pd.DataFrame()


def compute_indicators(df):
    df = df.copy()
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    if ta is not None:
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    else:
        df["RSI"] = 50

    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["VWAP_TOP"] = df["VWAP"] * 1.02
    df["VWAP_BOTTOM"] = df["VWAP"] * 0.98

    return df


# ============================================================
# SIMPLE DIVERGENCE ENGINE (PLACEHOLDER)
# ============================================================
def compute_divergences(df):
    # Placeholder: returns empty lists
    return [], []


# ============================================================
# SPARKLINE GENERATOR
# ============================================================
def generate_sparkline(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            line=dict(color="#D4AF37", width=2),
            hoverinfo="skip"
        ))

        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
            height=40,
            width=150,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    except:
        return None


# ============================================================
# TREND + DIVERGENCE BADGES
# ============================================================
def get_trend_badge(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            return "—"

        df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
        if df["Close"].iloc[-1] > df["EMA200"].iloc[-1]:
            return "<span style='color:#4caf50;font-weight:700;'>UPTREND</span>"
        else:
            return "<span style='color:#f44336;font-weight:700;'>DOWNTREND</span>"
    except:
        return "—"


def get_divergence_badge(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            return "—"

        df = compute_indicators(df)
        bull, bear = compute_divergences(df)

        if bull:
            return "<span style='color:#4caf50;font-weight:700;'>BULL</span>"
        if bear:
            return "<span style='color:#f44336;font-weight:700;'>BEAR</span>"
        return "<span style='color:#AAAAAA;'>NONE</span>"
    except:
        return "—"


# ============================================================
# SMC + ORDERFLOW + MTF + BIAS
# ============================================================
def detect_smc(df):
    smc = {
        "BOS": False,
        "CHOCH": False,
        "FVG": [],
        "LiquidityZones": []
    }
    if df.empty:
        return smc

    df = df.copy()
    df["HH"] = df["High"].rolling(3).max()
    df["LL"] = df["Low"].rolling(3).min()

    try:
        if df["Close"].iloc[-1] > df["HH"].iloc[-2]:
            smc["BOS"] = True
        if df["Close"].iloc[-1] < df["LL"].iloc[-2]:
            smc["CHOCH"] = True
    except:
        pass

    for i in range(2, len(df)):
        if df["Low"].iloc[i] > df["High"].iloc[i-2]:
            smc["FVG"].append(("Bullish", df.index[i]))
        if df["High"].iloc[i] < df["Low"].iloc[i-2]:
            smc["FVG"].append(("Bearish", df.index[i]))

        if abs(df["High"].iloc[i] - df["High"].iloc[i-1]) < 0.1 * df["High"].iloc[i]:
            smc["LiquidityZones"].append(("EQH", df.index[i]))
        if abs(df["Low"].iloc[i] - df["Low"].iloc[i-1]) < 0.1 * df["Low"].iloc[i]:
            smc["LiquidityZones"].append(("EQL", df.index[i]))

    return smc


def compute_orderflow(df):
    if df.empty:
        return "Neutral"
    df = df.copy()
    df["Delta"] = df["Close"] - df["Open"]
    df["Pressure"] = df["Delta"].rolling(10).sum()
    val = df["Pressure"].iloc[-1]
    if val > 0:
        return "Bullish"
    elif val < 0:
        return "Bearish"
    return "Neutral"


def get_mtf_divergences(ticker):
    timeframes = {
        "Weekly": ("1wk", 3),
        "Daily": ("1d", 1),
        "1H": ("1h", 0.25),
        "15M": ("15m", 0.1)
    }
    results = {}
    for tf_name, (tf, years) in timeframes.items():
        df = load_data(ticker, tf, years=years)
        if df.empty:
            results[tf_name] = {"last_bull": None, "last_bear": None}
            continue
        df = compute_indicators(df)
        bull, bear = compute_divergences(df)
        results[tf_name] = {
            "bull": bull,
            "bear": bear,
            "last_bull": bull[-1] if bull else None,
            "last_bear": bear[-1] if bear else None
        }
    return results


def compute_alignment_score(mtf):
    score = 0
    if mtf["Weekly"]["last_bull"]:
        score += 40
    if mtf["Weekly"]["last_bear"]:
        score -= 40
    if mtf["Daily"]["last_bull"]:
        score += 25
    if mtf["Daily"]["last_bear"]:
        score -= 25
    if mtf["1H"]["last_bull"]:
        score += 20
    if mtf["1H"]["last_bear"]:
        score -= 20
    if mtf["15M"]["last_bull"]:
        score += 15
    if mtf["15M"]["last_bear"]:
        score -= 15
    return max(min(score, 100), -100)


def compute_bias(alignment, orderflow, smc):
    score = alignment
    if orderflow == "Bullish":
        score += 15
    elif orderflow == "Bearish":
        score -= 15
    if smc["BOS"]:
        score += 20
    if smc["CHOCH"]:
        score -= 20

    if score > 40:
        return "Strong Bullish"
    if score > 10:
        return "Bullish"
    if score < -40:
        return "Strong Bearish"
    if score < -10:
        return "Bearish"
    return "Neutral"


# ============================================================
# AI TRADE IDEA GENERATOR (SIMPLIFIED)
# ============================================================
def generate_trade_idea(ticker, df, mtf, smc, orderflow):
    idea = {}
    df = df.copy()
    bull, bear = compute_divergences(df)

    if mtf["Daily"]["last_bull"] or bull:
        idea["Direction"] = "Long"
    elif mtf["Daily"]["last_bear"] or bear:
        idea["Direction"] = "Short"
    else:
        idea["Direction"] = "Neutral"

    idea["Entry"] = df["Close"].iloc[-1]

    if idea["Direction"] == "Long":
        idea["SL"] = df["Low"].rolling(10).min().iloc[-1]
        idea["TP"] = idea["Entry"] + 2 * (idea["Entry"] - idea["SL"])
    elif idea["Direction"] == "Short":
        idea["SL"] = df["High"].rolling(10).max().iloc[-1]
        idea["TP"] = idea["Entry"] - 2 * (idea["SL"] - idea["Entry"])
    else:
        idea["SL"] = None
        idea["TP"] = None

    risk = st.session_state["settings"]["default_risk"]
    if idea["SL"]:
        idea["PositionSize"] = round(risk / abs(idea["Entry"] - idea["SL"]), 2)
    else:
        idea["PositionSize"] = 0

    score = compute_alignment_score(mtf)
    if smc["BOS"]:
        score += 10
    if smc["CHOCH"]:
        score -= 10
    if orderflow == "Bullish":
        score += 10
    elif orderflow == "Bearish":
        score -= 10

    idea["Confluence"] = max(min(score, 100), -100)
    return idea


# ============================================================
# DIVERGENCE ALERT ENGINE
# ============================================================
def check_for_new_divergences(df_res):
    if "prev_signals" not in st.session_state:
        st.session_state["prev_signals"] = {}
    alerts = []
    for _, row in df_res.iterrows():
        t = row["Ticker"]
        sig = str(row["SignalDate"])
        if t not in st.session_state["prev_signals"]:
            st.session_state["prev_signals"][t] = sig
            continue
        if st.session_state["prev_signals"][t] != sig:
            alerts.append((t, sig))
            st.session_state["prev_signals"][t] = sig
    return alerts


def check_trend_change(ticker):
    if "trend_state" not in st.session_state:
        st.session_state["trend_state"] = {}
    badge = get_trend_badge(ticker)
    current = "UP" if "UPTREND" in badge else "DOWN"
    if ticker not in st.session_state["trend_state"]:
        st.session_state["trend_state"][ticker] = current
        return None
    if st.session_state["trend_state"][ticker] != current:
        st.session_state["trend_state"][ticker] = current
        return current
    return None


# ============================================================
# LUXURY SCREENER TABLE
# ============================================================
def render_screener_table(df_res, mode: str):
    df_show = df_res.copy()

    def strength_band(v):
        if v >= 150:
            return "Strong"
        elif v >= 80:
            return "Medium"
        else:
            return "Weak"

    df_show["Band"] = df_show["Strength"].apply(strength_band)

    if AGGRID_AVAILABLE:
        gb = GridOptionsBuilder.from_dataframe(
            df_show[["Ticker", "SignalDate", "Strength", "Band", "Bias"]] if mode == "Simple" else df_show
        )
        gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=15)
        gb.configure_side_bar()
        gb.configure_default_column(resizable=True, filter=True, sortable=True)
        gb.configure_column("Strength", cellStyle=lambda params: {
            "color": "black",
            "fontWeight": "700",
            "backgroundColor": (
                "#4caf50" if params.value >= 150 else
                "#ffc107" if params.value >= 80 else
                "#f44336"
            )
        })
        gb.configure_column("Band", cellStyle=lambda params: {
            "color": "black",
            "fontWeight": "600",
            "backgroundColor": (
                "#4caf50" if params.value == "Strong" else
                "#ffc107" if params.value == "Medium" else
                "#f44336"
            )
        })
        gb.configure_column("Bias", cellStyle=lambda p: {
            "color": "black",
            "fontWeight": "700",
            "backgroundColor": (
                "#4caf50" if "Bullish" in str(p.value) else
                "#f44336" if "Bearish" in str(p.value) else
                "#ffc107"
            )
        })
        grid_options = gb.build()
        AgGrid(
            df_show[["Ticker", "SignalDate", "Strength", "Band", "Bias"]] if mode == "Simple" else df_show,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.NO_UPDATE,
            theme="streamlit",
            height=400,
            fit_columns_on_grid_load=True,
        )
    else:
        styled = df_show.style.background_gradient(subset=["Strength"], cmap="Greens")
        if mode == "Simple":
            st.dataframe(styled[["Ticker", "SignalDate", "Strength", "Bias"]], use_container_width=True)
        else:
            st.dataframe(styled, use_container_width=True)


# ============================================================
# WATCHLIST PANEL (LUXURY)
# ============================================================
def render_watchlist_panel(df_res=None):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Watchlist")

    add_col, btn_col = st.columns([2, 1])
    with add_col:
        new_ticker = st.text_input("Add Ticker", value="", key="wl_add_ticker")
    with btn_col:
        if st.button("Add", key="wl_add_btn") and new_ticker.strip():
            t = new_ticker.strip().upper()
            if t not in st.session_state["watchlist"]:
                st.session_state["watchlist"].append(t)

    if st.button("Clear Watchlist", key="wl_clear_btn"):
        st.session_state["watchlist"] = []

    wl = st.session_state["watchlist"]
    if not wl:
        st.markdown("<p style='color:#AAAAAA;'>No symbols yet.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return None

    rows = []
    for t in wl:
        signal = None
        strength = None
        if df_res is not None and not df_res.empty and t in df_res["Ticker"].values:
            row = df_res[df_res["Ticker"] == t].iloc[0]
            signal = row["SignalDate"]
            strength = row["Strength"]
        rows.append({
            "Ticker": t,
            "SignalDate": signal,
            "Strength": strength,
            "Trend": get_trend_badge(t),
            "Divergence": get_divergence_badge(t)
        })

    df_wl = pd.DataFrame(rows)

    st.markdown("### Symbols")
    for _, r in df_wl.iterrows():
        st.markdown(
            f"""
            <div class="glass-card" style="padding:10px;margin-bottom:10px;">
                <h4 style="margin:0;color:#D4AF37;">{r['Ticker']}</h4>
                <p style="margin:0;">
                    Trend: {r['Trend']} • Divergence: {r['Divergence']}
                </p>
                <p style="margin:0;">
                    Strength: <b>{r['Strength']}</b> • Signal: {r['SignalDate']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        fig = generate_sparkline(r["Ticker"])
        if fig:
            st.plotly_chart(fig, use_container_width=True, height=40)

        trend_change = check_trend_change(r["Ticker"])
        if trend_change:
            notify(f"{r['Ticker']} trend changed to {trend_change}", "warning")

    selected = st.selectbox("Load Chart", df_wl["Ticker"], key="wl_chart_select")

    st.markdown('</div>', unsafe_allow_html=True)
    return selected


# ============================================================
# CHART FUNCTION
# ============================================================
def plot_ultra_pro_chart(df, bull_divs, bear_divs,
                         show_ema=True, show_avwap=True,
                         show_vwap_top=True, show_vwap_bottom=True,
                         show_div=True=True, show_rsi=True,
                         zoom_range="1Y"):
    df = df.copy()
    if df.empty:
        return go.Figure()

    if zoom_range == "1M":
        df = df.tail(22)
    elif zoom_range == "3M":
        df = df.tail(66)
    elif zoom_range == "6M":
        df = df.tail(132)
    elif zoom_range == "1Y":
        df = df.tail(252)

    fig = make_price_rsi_chart(df, show_rsi)

    if show_ema and "EMA200" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA200"], mode="lines",
            line=dict(color="#ff9800", width=1.5),
            name="EMA200", yaxis="y1"
        ))

    if show_avwap and "VWAP" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"], mode="lines",
            line=dict(color="#00bcd4", width=1.2),
            name="VWAP", yaxis="y1"
        ))

    if show_vwap_top and "VWAP_TOP" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP_TOP"], mode="lines",
            line=dict(color="#8bc34a", width=1, dash="dot"),
            name="VWAP Top", yaxis="y1"
        ))

    if show_vwap_bottom and "VWAP_BOTTOM" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP_BOTTOM"], mode="lines",
            line=dict(color="#f44336", width=1, dash="dot"),
            name="VWAP Bottom", yaxis="y1"
        ))

    # Divergences placeholder (no actual markers)
    return fig


def make_price_rsi_chart(df, show_rsi):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price"
    ))
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=600
    )
    return fig


# ============================================================
# SETTINGS PANEL
# ============================================================
def render_settings_panel():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Terminal Settings")

    s = st.session_state["settings"]

    st.markdown("### Theme Controls")
    s["gold_glow"] = st.slider("Gold Glow Intensity", 0.1, 1.0, s["gold_glow"], 0.05)

    st.markdown("### Chart Defaults")
    s["show_ema"] = st.checkbox("Show EMA200", s["show_ema"])
    s["show_avwap"] = st.checkbox("Show AVWAP", s["show_avwap"])
    s["show_vwap_top"] = st.checkbox("Show VWAP Top", s["show_vwap_top"])
    s["show_vwap_bottom"] = st.checkbox("Show VWAP Bottom", s["show_vwap_bottom"])
    s["show_divergences"] = st.checkbox("Show Divergences", s["show_divergences"])
    s["show_rsi"] = st.checkbox("Show RSI Panel", s["show_rsi"])

    st.markdown("### Screener Defaults")
    s["default_tf"] = st.selectbox(
        "Default Timeframe",
        ["1d", "1h", "15m", "1wk"],
        index=["1d", "1h", "15m", "1wk"].index(s["default_tf"])
    )

    st.markdown("### Backtest Defaults")
    s["default_risk"] = st.number_input("Default Risk per Trade (₹)", value=s["default_risk"])
    s["default_years"] = st.slider("Default Years of Data", 1, 5, s["default_years"])

    st.markdown("### Layout Mode")
    st.session_state["layout_mode"] = st.selectbox(
        "Choose Layout",
        ["Analyst", "Compact", "Dual Pane", "Full Screen Chart"],
        index=["Analyst", "Compact", "Dual Pane", "Full Screen Chart"].index(st.session_state["layout_mode"])
    )

    st.markdown("### Automation")
    st.session_state["auto_refresh"] = st.checkbox(
        "Enable Auto‑Refresh Screener (every 5 minutes)",
        st.session_state["auto_refresh"]
    )

    st.markdown("### Auto‑Trader Mode")
    st.session_state["auto_trader"]["enabled"] = st.checkbox(
        "Enable Auto‑Trader (Simulation Only)",
        st.session_state["auto_trader"]["enabled"]
    )

    st.markdown("### Presets")
    if st.button("Save Current Settings as Preset"):
        st.session_state["preset"] = s.copy()
        st.success("Preset saved.")
    if st.button("Load Preset") and "preset" in st.session_state:
        st.session_state["settings"] = st.session_state["preset"].copy()
        st.success("Preset loaded.")

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# AUTO-TRADER ENGINE (SIMULATED)
# ============================================================
def auto_execute_trade(ticker, idea):
    pos = st.session_state["auto_trader"]["positions"]
    if idea["Direction"] == "Neutral":
        return
    if idea["Direction"] == "Long":
        if ticker not in pos:
            pos[ticker] = {
                "side": "Long",
                "entry": idea["Entry"],
                "sl": idea["SL"],
                "tp": idea["TP"],
                "size": idea["PositionSize"]
            }
            st.session_state["auto_trader"]["trade_log"].append({
                "Ticker": ticker,
                "Action": "BUY",
                "Price": idea["Entry"],
                "Size": idea["PositionSize"],
                "Time": datetime.datetime.now()
            })
    if idea["Direction"] == "Short":
        if ticker not in pos:
            pos[ticker] = {
                "side": "Short",
                "entry": idea["Entry"],
                "sl": idea["SL"],
                "tp": idea["TP"],
                "size": idea["PositionSize"]
            }
            st.session_state["auto_trader"]["trade_log"].append({
                "Ticker": ticker,
                "Action": "SELL",
                "Price": idea["Entry"],
                "Size": idea["PositionSize"],
                "Time": datetime.datetime.now()
            })


def auto_exit_positions(df, ticker):
    pos = st.session_state["auto_trader"]["positions"]
    if ticker not in pos:
        return
    p = pos[ticker]
    price = df["Close"].iloc[-1]

    if p["side"] == "Long" and price <= p["sl"]:
        pnl = (price - p["entry"]) * p["size"]
        st.session_state["auto_trader"]["trade_log"].append({
            "Ticker": ticker,
            "Action": "EXIT_SL",
            "Price": price,
            "PnL": pnl,
            "Time": datetime.datetime.now()
        })
        del pos[ticker]

    if p["side"] == "Short" and price >= p["sl"]:
        pnl = (p["entry"] - price) * p["size"]
        st.session_state["auto_trader"]["trade_log"].append({
            "Ticker": ticker,
            "Action": "EXIT_SL",
            "Price": price,
            "PnL": pnl,
            "Time": datetime.datetime.now()
        })
        del pos[ticker]

    if p["side"] == "Long" and price >= p["tp"]:
        pnl = (price - p["entry"]) * p["size"]
        st.session_state["auto_trader"]["trade_log"].append({
            "Ticker": ticker,
            "Action": "EXIT_TP",
            "Price": price,
            "PnL": pnl,
            "Time": datetime.datetime.now()
        })
        del pos[ticker]

    if p["side"] == "Short" and price <= p["tp"]:
        pnl = (p["entry"] - price) * p["size"]
        st.session_state["auto_trader"]["trade_log"].append({
            "Ticker": ticker,
            "Action": "EXIT_TP",
            "Price": price,
            "PnL": pnl,
            "Time": datetime.datetime.now()
        })
        del pos[ticker]


def auto_rebalance():
    pos = st.session_state["auto_trader"]["positions"]
    n = len(pos)
    if n == 0:
        return
    equal_weight = 1 / n
    for t in pos:
        pos[t]["target_weight"] = equal_weight


def auto_hedge(ticker, df, bias):
    pos = st.session_state["auto_trader"]["positions"]
    if ticker not in pos:
        return
    if pos[ticker]["side"] == "Long" and "Bearish" in bias:
        pos[ticker]["hedged"] = True
    if pos[ticker]["side"] == "Short" and "Bullish" in bias:
        pos[ticker]["hedged"] = True


# ============================================================
# PORTFOLIO ANALYTICS (SIMPLIFIED)
# ============================================================
def compute_trade_stats(trades):
    if len(trades) == 0:
        return {}
    df = pd.DataFrame(trades)
    if "PnL" not in df.columns:
        return {}
    df["Return"] = df["PnL"] / df["Price"].replace(0, np.nan)
    df["Return"].fillna(0, inplace=True)

    win_rate = (df["PnL"] > 0).mean() * 100
    avg_win = df[df["PnL"] > 0]["PnL"].mean() if (df["PnL"] > 0).any() else 0
    avg_loss = df[df["PnL"] < 0]["PnL"].mean() if (df["PnL"] < 0).any() else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    equity = df["PnL"].cumsum()
    max_dd = (equity.cummax() - equity).max()

    if df["Return"].std() != 0:
        sharpe = df["Return"].mean() / df["Return"].std() * np.sqrt(252)
    else:
        sharpe = 0
    neg = df[df["Return"] < 0]["Return"]
    if neg.std() != 0:
        sortino = df["Return"].mean() / neg.std() * np.sqrt(252)
    else:
        sortino = 0

    return {
        "WinRate": win_rate,
        "ProfitFactor": profit_factor,
        "MaxDrawdown": max_dd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "TotalPnL": df["PnL"].sum()
    }


def simulate_equity_curve(trades):
    if len(trades) == 0:
        return None
    df = pd.DataFrame(trades)
    if "PnL" not in df.columns or "Time" not in df.columns:
        return None
    df["Equity"] = df["PnL"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Time"], y=df["Equity"],
        mode="lines", line=dict(color="#D4AF37", width=3)
    ))
    fig.update_layout(
        title="Equity Curve",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300
    )
    return fig


# ============================================================
# MAIN APP
# ============================================================
def main():
    st.set_page_config(page_title="Sniper Divergence Terminal", layout="wide")

    st.markdown(
        """
        <style>
        .glass-card {
            background: rgba(15, 15, 30, 0.85);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            border: 1px solid rgba(212,175,55,0.35);
            box-shadow: 0 0 18px rgba(212,175,55,0.25);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Sniper Divergence Terminal – Ultra Pro")

    tab_screener, tab_backtest, tab_settings, tab_portfolio, tab_autotrader = st.tabs(
        ["📊 Screener", "📈 Backtest", "⚙️ Settings", "📚 Portfolio Analytics", "🤖 Auto‑Trader"]
    )

    # ========================================================
    # SCREENER TAB
    # ========================================================
    with tab_screener:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.title("Sniper Divergence Screener – NIFTY200")

        col1, col2, col3 = st.columns(3)
        with col1:
            universe = st.selectbox("Universe", ["NIFTY200", "Custom"], key="universe_sel")
        with col2:
            tf_default = st.session_state["settings"]["default_tf"]
            interval_s = st.selectbox(
                "Timeframe", ["1d", "1h", "15m", "1wk"],
                index=["1d", "1h", "15m", "1wk"].index(tf_default),
                key="screener_tf"
            )
        with col3:
            mode_view = st.selectbox("View Mode", ["Simple", "Detailed"], key="view_mode_sel")

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

        if universe == "Custom":
            custom = st.text_input("Enter tickers", "HAL.NS", key="custom_tickers")
            tickers = [x.strip() for x in custom.split(",") if x.strip()]
        else:
            tickers = NIFTY200

        run_col, info_col = st.columns([1, 3])
        with run_col:
            run_click = st.button("Run Screener", key="run_screener_btn")
        with info_col:
            st.markdown(
                "<span style='color:#D4AF37;font-weight:600;'>Wall Street Grid • Metallic Gold Mode</span>",
                unsafe_allow_html=True
            )

        # Auto-refresh engine
        if st.session_state["auto_refresh"]:
            now = datetime.datetime.now()
            if st.session_state["last_scan"] is None or (now - st.session_state["last_scan"]).seconds > 300:
                run_click = True
                st.session_state["last_scan"] = now
                notify("Auto‑refresh: scanning universe…", "info")

        if run_click:
            st.info("Scanning universe for bullish divergences…")
            results = []
            for t in tickers:
                df = load_data(t, interval_s, years=1)
                if df.empty:
                    continue
                df = compute_indicators(df)
                bull, bear = compute_divergences(df)
                if not bull and not bear:
                    continue
                strength = np.random.randint(50, 200)
                signal_date = df.index[-1]
                mtf = get_mtf_divergences(t)
                alignment = compute_alignment_score(mtf)
                smc = detect_smc(df)
                orderflow = compute_orderflow(df)
                bias = compute_bias(alignment, orderflow, smc)
                results.append({
                    "Ticker": t,
                    "SignalDate": signal_date,
                    "Strength": strength,
                    "Alignment": alignment,
                    "Bias": bias
                })
            if not results:
                st.warning("No setups found.")
            else:
                st.session_state["screener_results"] = pd.DataFrame(results).sort_values("Strength", ascending=False)

        st.markdown('</div>', unsafe_allow_html=True)

        df_res = st.session_state["screener_results"]
        if df_res is not None and not df_res.empty:
            left_col, right_col = st.columns([3, 1])

            with left_col:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Signals Grid")
                render_screener_table(df_res, mode_view)

                selected = st.selectbox(
                    "Select a ticker to view chart",
                    df_res["Ticker"],
                    key="chart_ticker_select"
                )
                st.markdown('</div>', unsafe_allow_html=True)

                if selected:
                    row = df_res[df_res["Ticker"] == selected].iloc[0]
                    ticker = row["Ticker"]
                    df = load_data(ticker, interval_s, years=st.session_state["settings"]["default_years"])
                    df = compute_indicators(df)
                    bull_divs, bear_divs = compute_divergences(df)

                    st.markdown(
                        f"""
                        <div class="glass-card">
                            <h4 style="margin-top:0;">{ticker} • Divergence Snapshot</h4>
                            <p>Last Signal: <b>{row['SignalDate'].strftime('%Y-%m-%d')}</b> • 
                            Strength: <b>{row['Strength']}</b> • Bias: <b>{row['Bias']}</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("Chart Controls")
                    c1, c2, c3, c4, c5, c6 = st.columns(6)
                    s = st.session_state["settings"]
                    with c1:
                        s["show_ema"] = st.toggle("EMA200", s["show_ema"])
                    with c2:
                        s["show_avwap"] = st.toggle("AVWAP", s["show_avwap"])
                    with c3:
                        s["show_vwap_top"] = st.toggle("VWAP Top", s["show_vwap_top"])
                    with c4:
                        s["show_vwap_bottom"] = st.toggle("VWAP Bottom", s["show_vwap_bottom"])
                    with c5:
                        s["show_divergences"] = st.toggle("Divergences", s["show_divergences"])
                    with c6:
                        s["show_rsi"] = st.toggle("RSI Panel", s["show_rsi"])

                    st.markdown("### Zoom Presets")
                    z1, z2, z3, z4 = st.columns(4)
                    if "zoom_range" not in st.session_state:
                        st.session_state["zoom_range"] = "1Y"
                    if z1.button("1M"):
                        st.session_state["zoom_range"] = "1M"
                    if z2.button("3M"):
                        st.session_state["zoom_range"] = "3M"
                    if z3.button("6M"):
                        st.session_state["zoom_range"] = "6M"
                    if z4.button("1Y"):
                        st.session_state["zoom_range"] = "1Y"

                    if st.button("📸 Download Chart as PNG"):
                        st.session_state["download_chart"] = True

                    st.markdown('</div>', unsafe_allow_html=True)

                    with st.expander(f"📈 Full-Screen Chart – {ticker}", expanded=True):
                        fig = plot_ultra_pro_chart(
                            df, bull_divs, bear_divs,
                            s["show_ema"], s["show_avwap"],
                            s["show_vwap_top"], s["show_vwap_bottom"],
                            s["show_divergences"], s["show_rsi"],
                            st.session_state["zoom_range"]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        if st.session_state.get("download_chart", False):
                            try:
                                png = fig.to_image(format="png")
                                st.download_button(
                                    label="Download PNG",
                                    data=png,
                                    file_name=f"{ticker}_chart.png",
                                    mime="image/png"
                                )
                            except Exception:
                                st.warning("Plotly image export not available in this environment.")
                            st.session_state["download_chart"] = False

                    # AI Trade Idea
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("AI Trade Idea")
                    mtf = get_mtf_divergences(ticker)
                    alignment = compute_alignment_score(mtf)
                    smc = detect_smc(df)
                    orderflow = compute_orderflow(df)
                    bias = compute_bias(alignment, orderflow, smc)
                    idea = generate_trade_idea(ticker, df, mtf, smc, orderflow)

                    st.write(f"**Direction:** {idea['Direction']}")
                    st.write(f"**Entry:** {idea['Entry']}")
                    st.write(f"**Stop Loss:** {idea['SL']}")
                    st.write(f"**Take Profit:** {idea['TP']}")
                    st.write(f"**Position Size:** {idea['PositionSize']} units")
                    st.write(f"**Confluence Score:** {idea['Confluence']} / 100")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Auto-trader integration
                    if st.session_state["auto_trader"]["enabled"]:
                        auto_execute_trade(ticker, idea)
                        auto_exit_positions(df, ticker)
                        auto_hedge(ticker, df, bias)
                        auto_rebalance()

            with right_col:
                wl_selected = render_watchlist_panel(df_res)
                if wl_selected:
                    df_w = load_data(wl_selected, interval_s, years=1)
                    df_w = compute_indicators(df_w)
                    bull_w, bear_w = compute_divergences(df_w)
                    fig_w = plot_ultra_pro_chart(
                        df_w, bull_w, bear_w,
                        st.session_state["settings"]["show_ema"],
                        st.session_state["settings"]["show_avwap"],
                        st.session_state["settings"]["show_vwap_top"],
                        st.session_state["settings"]["show_vwap_bottom"],
                        st.session_state["settings"]["show_divergences"],
                        st.session_state["settings"]["show_rsi"],
                        st.session_state.get("zoom_range", "1Y")
                    )
                    st.plotly_chart(fig_w, use_container_width=True)

            alerts = check_for_new_divergences(df_res)
            for t, sig in alerts:
                notify(f"New divergence detected on {t} at {sig}", "success")

    # ========================================================
    # BACKTEST TAB (SIMPLE PLACEHOLDER)
    # ========================================================
    with tab_backtest:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Backtest (Placeholder)")
        st.info("Hook your existing backtest engine here and push trades into st.session_state['trade_history'].")
        st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================
    # SETTINGS TAB
    # ========================================================
    with tab_settings:
        render_settings_panel()

    # ========================================================
    # PORTFOLIO ANALYTICS TAB
    # ========================================================
    with tab_portfolio:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Portfolio Analytics")
        trades = st.session_state["auto_trader"]["trade_log"]
        if not trades:
            st.info("No trades yet. Auto‑trader or backtest must generate trades.")
        else:
            stats = compute_trade_stats(trades)
            st.write("### Trade Statistics")
            st.write(stats)
            fig_eq = simulate_equity_curve(trades)
            if fig_eq:
                st.plotly_chart(fig_eq, use_container_width=True)
            st.write("### Trade Log")
            st.dataframe(pd.DataFrame(trades), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================
    # AUTO-TRADER DASHBOARD
    # ========================================================
    with tab_autotrader:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Auto‑Trader Dashboard")
        st.write("### Open Positions")
        st.write(st.session_state["auto_trader"]["positions"])
        st.write("### Trade Log")
        st.dataframe(pd.DataFrame(st.session_state["auto_trader"]["trade_log"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
