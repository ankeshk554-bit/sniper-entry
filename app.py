import warnings
warnings.filterwarnings("ignore")

import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# =========================================================
# UNIVERSE
# =========================================================
NIFTY50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "HINDUNILVR.NS", "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS",
    "KOTAKBANK.NS", "HCLTECH.NS", "ASIANPAINT.NS", "MARUTI.NS", "AXISBANK.NS",
    "SUNPHARMA.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS", "WIPRO.NS", "TITAN.NS",
    "ONGC.NS", "POWERGRID.NS", "NTPC.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "M&M.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "BRITANNIA.NS", "NESTLEIND.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "BAJAJ-AUTO.NS", "COALINDIA.NS", "GRASIM.NS",
    "TECHM.NS", "CIPLA.NS", "BPCL.NS", "IOC.NS", "HINDALCO.NS",
    "TATACONSUM.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "INDUSINDBK.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "DMART.NS",
]

NIFTY200 = NIFTY50 + [
    "ABB.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "AUROPHARMA.NS",
    "BANDHANBNK.NS", "BANKBARODA.NS", "BEL.NS", "BERGEPAINT.NS",
    "BIOCON.NS", "BOSCHLTD.NS", "CANBK.NS", "CHOLAFIN.NS", "CUMMINSIND.NS",
    "DABUR.NS", "DLF.NS", "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS",
    "ICICIPRULI.NS", "IGL.NS", "INDHOTEL.NS", "INDIGO.NS", "INDUSINDBK.NS",
    "LUPIN.NS", "MFSL.NS", "MUTHOOTFIN.NS", "NAUKRI.NS", "PIDILITIND.NS",
    "PNB.NS", "POLYCAB.NS", "RECLTD.NS", "SAIL.NS", "SRF.NS",
    "TATACONSUM.NS", "TRENT.NS", "TVSMOTOR.NS", "VOLTAS.NS", "HAL.NS",
    "IRCTC.NS", "ZOMATO.NS", "LTIM.NS", "VEDL.NS", "UPL.NS", "ZEEL.NS",
]
NIFTY200 = list(dict.fromkeys(NIFTY200))

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Sniper Terminal v2 - Ankesh",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# CSS
# =========================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
body, .stApp {
    background: #07090f !important;
    color: #c9d1d9 !important;
}
.stApp {
    font-family: "JetBrains Mono", monospace !important;
}
#MainMenu, footer, header {visibility:hidden;}
/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-bottom: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] {
    font-family: "JetBrains Mono", monospace;
    color: #8b949e;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff;
}
/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: #ffffff !important;
    border: none !important;
    font-family: "JetBrains Mono", monospace !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border-radius: 6px !important;
    padding: 9px 24px !important;
    transition: all .2s !important;
}
.stButton>button:hover {
    filter: brightness(1.15) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(35, 134, 54, .4) !important;
}
.stButton>button:disabled {
    opacity: .4 !important;
    transform: none !important;
}
/* Inputs */
.stTextInput>div>div>input,
.stNumberInput>div>div>input {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
    font-family: "JetBrains Mono", monospace !important;
}
.stSelectbox>div>div {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
}
/* Toggle */
.stCheckbox, .stToggle {
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    color: #c9d1d9;
}
/* Dataframe */
.stDataFrame {
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
}
/* Metrics */
[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
}
[data-testid="stMetricValue"] {
    font-family: "JetBrains Mono", monospace;
    font-size: 20px;
}
/* Expander */
.streamlit-expanderHeader {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
}
/* Sidebar */
.css-1d391kg {
    background: #07090f;
}
/* Progress bar */
.stProgress>div>div>div {
    background: linear-gradient(90deg, #238636, #2ea043) !important;
}
/* Info/Warning */
.stInfo {
    background: rgba(88, 166, 255, .08) !important;
    border: 1px solid rgba(88, 166, 255, .4) !important;
}
.stWarning {
    background: rgba(210, 153, 34, .08) !important;
    border: 1px solid rgba(210, 153, 34, .4) !important;
}
.stSuccess {
    background: rgba(35, 134, 54, .08) !important;
    border: 1px solid rgba(35, 134, 54, .4) !important;
}
.stError {
    background: rgba(218, 54, 51, .08) !important;
    border: 1px solid rgba(218, 54, 51, .4) !important;
}
/* Scrollbar */
::-webkit-scrollbar {width: 4px; height: 4px;}
::-webkit-scrollbar-track {background: #07090f;}
::-webkit-scrollbar-thumb {background: #30363d; border-radius: 4px;}
/* Cards */
.sig-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 12px 14px;
}
.sig-card:hover { border-color: #58a6ff; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================================================
# DATA LOADER
# =========================================================
@st.cache_data(show_spinner=False, ttl=300)
def load_data(ticker: str, interval: str, years: int = 1) -> pd.DataFrame:
    period_map = {
        "1wk": "7y",
        "1d": f"{years}y",
        "1h": f"{min(years, 2)}y",
        "15m": f"{min(years, 1)}y",
    }
    df = yf.download(
        ticker,
        period=period_map.get(interval, f"{years}y"),
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            return pd.DataFrame()
    return df

# =========================================================
# INDICATORS
# =========================================================
@st.cache_data(show_spinner=False)
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, hi, lo, v = df["Close"], df["High"], df["Low"], df["Volume"]

    for p in [9, 21, 50, 200]:
        df[f"EMA{p}"] = c.ewm(span=p, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = e12 - e26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    tr = pd.concat(
        [hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.ewm(span=14, adjust=False).mean()

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["BB_U"] = sma20 + 2 * std20
    df["BB_M"] = sma20
    df["BB_L"] = sma20 - 2 * std20
    df["BB_W"] = (df["BB_U"] - df["BB_L"]) / df["BB_M"].replace(0, np.nan)

    ema20 = c.ewm(span=20, adjust=False).mean()
    df["KC_U"] = ema20 + 1.5 * df["ATR"]
    df["KC_L"] = ema20 - 1.5 * df["ATR"]
    df["Squeeze"] = (df["BB_U"] < df["KC_U"]) & (df["BB_L"] > df["KC_L"])
    df["Squeeze_Fire"] = df["Squeeze"].shift(1).fillna(False) & ~df["Squeeze"]

    lo14 = lo.rolling(14).min()
    hi14 = hi.rolling(14).max()
    df["Stoch_K"] = 100 * (c - lo14) / (hi14 - lo14).replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    tp = (hi + lo + c) / 3
    df["AVWAP"] = (tp * v).cumsum() / v.cumsum()
    var = ((tp - df["AVWAP"]) ** 2 * v).cumsum() / v.cumsum()
    sd = np.sqrt(var.clip(lower=0))
    df["VWAP_U1"] = df["AVWAP"] + sd
    df["VWAP_L1"] = df["AVWAP"] - sd
    df["VWAP_U2"] = df["AVWAP"] + 2 * sd
    df["VWAP_L2"] = df["AVWAP"] - 2 * sd

    df["Vol_MA20"] = v.rolling(20).mean()
    df["Vol_Ratio"] = v / df["Vol_MA20"].replace(0, np.nan)

    body = (c - df["Open"]).abs()
    rng = (hi - lo).replace(0, np.nan)
    br = (body / rng).clip(0, 1)
    bull_b = (c > df["Open"]).astype(float)
    df["OFI"] = (bull_b * br - (1 - bull_b) * br).rolling(5).mean()

    bdy = (c - df["Open"]).abs()
    lwck = df[["Open", "Close"]].min(axis=1) - lo
    uwck = hi - df[["Open", "Close"]].max(axis=1)
    df["Hammer"] = (lwck > 2 * bdy) & (uwck < bdy * 0.5) & (c > df["Open"])

    prev_bear = c.shift(1) < df["Open"].shift(1)
    curr_bull = c > df["Open"]
    df["BullEngulf"] = (
        prev_bear
        & curr_bull
        & (df["Open"] <= c.shift(1))
        & (c >= df["Open"].shift(1))
    )
    df["BearEngulf"] = (
        (c.shift(1) > df["Open"].shift(1))
        & (c < df["Open"])
        & (df["Open"] >= c.shift(1))
    )

    df["BullPat"] = df["Hammer"] | df["BullEngulf"]
    df["BearPat"] = df["BearEngulf"] | (
        (uwck > 2 * bdy) & (lwck < bdy * 0.5) & (c < df["Open"])
    )
    return df

# =========================================================
# AVWAPS
# =========================================================
@st.cache_data(show_spinner=False)
def compute_avwaps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"]

    def avwap(anc: int) -> pd.Series:
        out = pd.Series(np.nan, index=df.index)
        if anc >= len(df):
            return out
        cv = vol.iloc[anc:].cumsum()
        ctv = (tp.iloc[anc:] * vol.iloc[anc:]).cumsum()
        out.iloc[anc:] = (ctv / cv.replace(0, np.nan)).values
        return out

    tl = int(df["High"].values.argmax())
    bl = int(df["Low"].values.argmin())
    rec = max(0, len(df) - 60)
    rt = rec + int(df["High"].iloc[rec:].values.argmax())
    rb = rec + int(df["Low"].iloc[rec:].values.argmin())

    df["VWAP_TOP"] = avwap(tl)
    df["VWAP_BOT"] = avwap(bl)
    df["VWAP_RTOP"] = avwap(rt)
    df["VWAP_RBOT"] = avwap(rb)
    return df

# =========================================================
# DIVERGENCES
# =========================================================
def swing_lows(series: pd.Series, bars: int = 5) -> np.ndarray:
    v = series.values
    m = np.zeros(len(v), dtype=bool)
    for i in range(bars, len(v) - bars):
        if v[i] == v[i - bars : i + bars + 1].min():
            m[i] = True
    return m


def swing_highs(series: pd.Series, bars: int = 5) -> np.ndarray:
    v = series.values
    m = np.zeros(len(v), dtype=bool)
    for i in range(bars, len(v) - bars):
        if v[i] == v[i - bars : i + bars + 1].max():
            m[i] = True
    return m


@st.cache_data(show_spinner=False)
def compute_divergences(df: pd.DataFrame, bars: int = 5):
    lm = swing_lows(df["Low"], bars)
    hm = swing_highs(df["High"], bars)
    li = np.where(lm)[0]
    hi = np.where(hm)[0]

    bull, bear = [], []

    for j in range(1, len(li)):
        i1, i2 = li[j - 1], li[j]
        if (
            df["Low"].iloc[i2] < df["Low"].iloc[i1]
            and df["RSI"].iloc[i2] > df["RSI"].iloc[i1]
            and df["MACD_Hist"].iloc[i2] > df["MACD_Hist"].iloc[i1]
        ):
            bull.append((i1, i2))

    for j in range(1, len(hi)):
        i1, i2 = hi[j - 1], hi[j]
        if (
            df["High"].iloc[i2] > df["High"].iloc[i1]
            and df["RSI"].iloc[i2] < df["RSI"].iloc[i1]
            and df["MACD_Hist"].iloc[i2] < df["MACD_Hist"].iloc[i1]
        ):
            bear.append((i1, i2))

    return bull[-3:], bear[-3:]

# =========================================================
# SIGNAL QUALITY
# =========================================================
def signal_quality(df: pd.DataFrame, i2: int):
    sc = {}
    cl = float(df["Close"].iloc[i2])
    atr = float(df["ATR"].iloc[i2])
    rsi = float(df["RSI"].iloc[i2])

    sc["RSI_Zone"] = 10 if 28 < rsi < 48 else 6 if rsi < 55 else 2

    mh = float(df["MACD_Hist"].iloc[i2])
    mhp = float(df["MACD_Hist"].iloc[i2 - 1]) if i2 > 0 else mh
    sc["MACD_Turn"] = 10 if mh > mhp and mh > -atr * 0.01 else 4

    e21 = float(df["EMA21"].iloc[i2])
    e50 = float(df["EMA50"].iloc[i2])
    e200 = float(df["EMA200"].iloc[i2]) if "EMA200" in df.columns else e50
    sc["EMA_Stack"] = 10 if cl > e21 > e50 > e200 else 6 if cl > e50 > e200 else 2

    sc["AVWAP"] = 10 if cl > float(df["AVWAP"].iloc[i2]) else 3

    sqf = bool(df["Squeeze_Fire"].iloc[i2])
    sqn = bool(df["Squeeze"].iloc[i2])
    sc["BB_Squeeze"] = 10 if sqf else 6 if sqn else 1

    vr = float(df["Vol_Ratio"].iloc[i2]) if not np.isnan(df["Vol_Ratio"].iloc[i2]) else 1.0
    sc["Vol_Surge"] = 10 if vr > 2 else 7 if vr > 1.5 else 2

    ofi = float(df["OFI"].iloc[i2]) if not np.isnan(df["OFI"].iloc[i2]) else 0.0
    sc["Order_Flow"] = 10 if ofi > 0.2 else 5 if ofi > 0 else 1

    sc["Candle_Pat"] = 10 if bool(df["BullPat"].iloc[i2]) else 3

    stk = float(df["Stoch_K"].iloc[i2]) if not np.isnan(df["Stoch_K"].iloc[i2]) else 50.0
    sc["Stochastic"] = 10 if stk < 25 else 5 if stk < 40 else 1

    sc["ATR_Valid"] = 10 if atr > 0 else 0

    tot = sum(sc.values())
    if tot >= 85:
        gr = "A+"
    elif tot >= 70:
        gr = "A"
    elif tot >= 55:
        gr = "B+"
    elif tot >= 40:
        gr = "B"
    else:
        gr = "C"

    return {"total": tot, "grade": gr, "breakdown": sc}

# =========================================================
# WEEKLY TREND
# =========================================================
@st.cache_data(show_spinner=False)
def get_weekly_trend(ticker: str):
    df = load_data(ticker, "1wk", years=5)
    if df.empty:
        return None
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    d = df["Close"].diff()
    g = d.clip(lower=0).ewm(span=14, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + g / l.replace(0, np.nan))
    return (df["Close"] > df["EMA200"]) & (df["RSI"] > 50)

# =========================================================
# SCANNER
# =========================================================
def scan_stock(ticker, interval, use_trend, fresh_only, bars, min_q):
    try:
        df = load_data(ticker, interval)
        if df.empty or len(df) < 100:
            return None
        df = compute_indicators(df)
        bull, _ = compute_divergences(df, bars=bars)
        if not bull:
            return None
        i1, i2 = bull[-1]
        if fresh_only and i2 < len(df) - 4:
            return None

        if use_trend:
            tw = get_weekly_trend(ticker)
            if tw is None:
                return None
            if not bool(tw.reindex(df.index, method="ffill").iloc[i2]):
                return None

        ei = i2 + 1
        if ei >= len(df):
            return None

        ep = float(df["Open"].iloc[ei])
        atr = float(df["ATR"].iloc[i2])
        e200 = float(df["EMA200"].iloc[i2]) if "EMA200" in df.columns else ep
        avwap = float(df["AVWAP"].iloc[i2])

        if ep < e200 or ep < avwap or atr <= 0:
            return None

        q = signal_quality(df, i2)
        if q["total"] < min_q:
            return None

        sl = round(ep - 1.5 * atr, 2)
        tp1 = round(ep + 2 * atr, 2)
        tp2 = round(ep + 4 * atr, 2)
        rr = round((tp1 - ep) / (ep - sl), 2) if ep != sl else 0

        vr = float(df["Vol_Ratio"].iloc[i2]) if not np.isnan(df["Vol_Ratio"].iloc[i2]) else 1.0
        ofi = float(df["OFI"].iloc[i2]) if not np.isnan(df["OFI"].iloc[i2]) else 0.0

        return {
            "Ticker": ticker,
            "Date": str(df.index[ei])[:10],
            "Entry": ep,
            "SL": sl,
            "TP1": tp1,
            "TP2": tp2,
            "R_R": rr,
            "Quality": q["total"],
            "Grade": q["grade"],
            "RSI": round(float(df["RSI"].iloc[i2]), 1),
            "Vol_Ratio": round(vr, 2),
            "Squeeze_Fire": bool(df["Squeeze_Fire"].iloc[i2]),
            "In_Squeeze": bool(df["Squeeze"].iloc[i2]),
            "OFI": round(ofi, 3),
            "Bull_Pat": bool(df["BullPat"].iloc[i2]),
            "ATR": round(atr, 2),
            "i1": i1,
            "i2": i2,
        }
    except Exception:
        return None

# =========================================================
# BACKTEST ENGINE
# =========================================================
def run_backtest(df, bull_divs, risk, trend_s, use_trend, trail=2.0):
    df = df.reset_index(drop=True)

    if use_trend and trend_s is not None:
        try:
            ts = trend_s.reindex(range(len(df)), method="ffill")
        except Exception:
            ts = pd.Series(True, index=range(len(df)))
    else:
        ts = pd.Series(True, index=range(len(df)))

    trades = []
    eq = [100.0]

    for i1, i2 in bull_divs:
        if not bool(ts.iloc[i2] if i2 < len(ts) else True):
            continue

        ei = i2 + 1
        if ei >= len(df) - 1:
            continue

        ep = float(df["Open"].iloc[ei])
        atr = float(df["ATR"].iloc[i2])
        e200 = float(df["EMA200"].iloc[i2]) if "EMA200" in df.columns else ep
        avwap = float(df["AVWAP"].iloc[i2])

        if ep < e200 or ep < avwap or atr <= 0:
            continue

        sl0 = ep - 1.5 * atr
        tp1 = ep + 2 * atr
        tp2 = ep + 4 * atr
        rpp = ep - sl0
        if rpp <= 0:
            continue

        qty = max(int(risk / rpp), 1)
        csl = sl0
        pdone = False
        xp = None
        xi = None
        xr = "TIME"
        rem = qty

        for j in range(ei + 1, min(ei + 20, len(df))):
            hi = float(df["High"].iloc[j])
            lo = float(df["Low"].iloc[j])
            cl = float(df["Close"].iloc[j])

            csl = max(csl, cl - trail * float(df["ATR"].iloc[j]))

            if lo <= csl:
                xp, xi, xr = csl, j, "SL"
                break

            if not pdone and hi >= tp1:
                pdone = True
                half = qty // 2
                pnl1 = (tp1 - ep) * half
                trades.append(
                    {
                        "Entry_Date": str(df.index[ei])[:10],
                        "Exit_Date": str(df.index[j])[:10],
                        "Entry": round(ep, 2),
                        "Exit": round(tp1, 2),
                        "Qty": half,
                        "PnL": round(pnl1, 2),
                        "Reason": "TP1",
                        "Ret_Pct": round((tp1 - ep) / ep * 100, 2),
                    }
                )
                eq.append(max(eq[-1] * (1 + pnl1 / (risk * 20)), 0.01))
                rem = qty - half

            if hi >= tp2:
                xp, xi, xr = tp2, j, "TP2"
                break

        if xp is None:
            li = min(ei + 19, len(df) - 1)
            xp = float(df["Close"].iloc[li])
            xi = li

        pnl = (xp - ep) * rem
        ret = (xp - ep) / ep * 100
        trades.append(
            {
                "Entry_Date": str(df.index[ei])[:10],
                "Exit_Date": str(df.index[xi])[:10],
                "Entry": round(ep, 2),
                "Exit": round(xp, 2),
                "Qty": rem,
                "PnL": round(pnl, 2),
                "Reason": xr,
                "Ret_Pct": round(ret, 2),
            }
        )
        eq.append(max(eq[-1] * (1 + pnl / (risk * 20)), 0.01))

    if not trades:
        return pd.DataFrame(), {}

    dft = pd.DataFrame(trades)
    rets = dft["Ret_Pct"].values / 100
    wins = rets[rets > 0]
    losses = rets[rets < 0]

    std = rets.std() if len(rets) > 1 else 1e-9
    neg_std = losses.std() if len(losses) > 1 else 1e-9

    sharpe = round(rets.mean() / std * (252 ** 0.5), 2) if std else 0
    sortino = round(rets.mean() / neg_std * (252 ** 0.5), 2) if neg_std else 0

    ea = np.array(eq)
    pk = np.maximum.accumulate(ea)
    dd = (pk - ea) / pk * 100
    max_dd = float(dd.max())
    calmar = round((ea[-1] - ea[0]) / ea[0] * 100 / max(max_dd, 0.01), 2)

    wr = round(len(wins) / len(rets) * 100, 1) if len(rets) else 0
    aw = round(wins.mean() * 100, 2) if len(wins) else 0
    al = round(abs(losses.mean()) * 100, 2) if len(losses) else 0
    pf = round(wins.sum() / abs(losses.sum()), 2) if len(losses) and losses.sum() != 0 else 0

    p = len(wins) / len(rets) if len(rets) else 0.5
    a = abs(losses.mean()) if len(losses) else 1e-9
    b = wins.mean() if len(wins) else 1e-9
    kelly = round(max(0, p / a - (1 - p) / b) * 100, 1) if a > 0 else 0

    return dft, {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Win_Rate": wr,
        "Avg_Win": aw,
        "Avg_Loss": al,
        "Profit_Factor": pf,
        "Max_DD": round(max_dd, 2),
        "Total_PnL": round(dft["PnL"].sum(), 2),
        "Total_Return": round((ea[-1] - ea[0]) / ea[0] * 100, 2),
        "Trades": len(dft),
        "Kelly": kelly,
        "Equity": eq,
    }

# =========================================================
# AUTO TRADE ENGINE (PAPER + SHOONYA STUB)
# =========================================================
def get_autotrade_state():
    if "at_active" not in st.session_state:
        st.session_state["at_active"] = False
    if "at_log" not in st.session_state:
        st.session_state["at_log"] = []
    if "at_pos" not in st.session_state:
        st.session_state["at_pos"] = {}
    if "at_pnl" not in st.session_state:
        st.session_state["at_pnl"] = 0.0
    if "at_mode" not in st.session_state:
        st.session_state["at_mode"] = "paper"


def paper_place_order(ticker, direction, qty, price, sl, tp1, reason):
    ts = time.strftime("%H:%M:%S")
    entry = {
        "time": ts,
        "ticker": ticker,
        "dir": direction,
        "qty": qty,
        "entry": price,
        "sl": sl,
        "tp1": tp1,
        "status": "OPEN",
        "pnl": 0.0,
        "reason": reason,
    }
    st.session_state["at_pos"][ticker] = entry
    st.session_state["at_log"].append(
        f"[{ts}] PAPER {direction} {qty}x {ticker} @ {price:.2f} | SL: {sl:.2f} TP1: {tp1:.2f} | {reason}"
    )


def paper_check_exits(risk_per_trade):
    to_close = []
    for tk, pos in st.session_state["at_pos"].items():
        if pos["status"] != "OPEN":
            continue
        try:
            df = load_data(tk, "1d", years=1)
            if df.empty:
                continue
            last = df.iloc[-1]
            hi = float(last["High"])
            lo = float(last["Low"])
            if lo <= pos["sl"]:
                pnl = (pos["sl"] - pos["entry"]) * pos["qty"]
                st.session_state["at_pnl"] += pnl
                st.session_state["at_log"].append(
                    f"[EXIT-SL] {tk} @ {pos['sl']:.2f} | PnL: INR {pnl:.2f}"
                )
                to_close.append(tk)
            elif hi >= pos["tp1"]:
                pnl = (pos["tp1"] - pos["entry"]) * pos["qty"]
                st.session_state["at_pnl"] += pnl
                st.session_state["at_log"].append(
                    f"[EXIT-TP] {tk} @ {pos['tp1']:.2f} | PnL: INR {pnl:.2f}"
                )
                to_close.append(tk)
        except Exception:
            pass
    for tk in to_close:
        st.session_state["at_pos"][tk]["status"] = "CLOSED"


def shoonya_place_order(ticker, direction, qty, price):
    # Stub: replace with actual Shoonya API call
    # from api_helper import ShoonyaApiPy
    # api.place_order(...)
    return {"status": "ok", "msg": "Shoonya stub - connect api_helper.py"}

# =========================================================
# CHARTS
# =========================================================
def plot_chart(df, bull_divs, bear_divs, ticker):
    df = compute_avwaps(df)
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.14, 0.18, 0.18],
        vertical_spacing=0.02,
        subplot_titles=[ticker, "Volume / OFI", "RSI (14)", "MACD"],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#3fb950",
            decreasing_line_color="#f85149",
            increasing_fillcolor="#3fb950",
            decreasing_fillcolor="#f85149",
        ),
        row=1,
        col=1,
    )

    for sp, col, nm in [
        (21, "#e3b341", "EMA21"),
        (50, "#58a6ff", "EMA50"),
        (200, "#ff9a3c", "EMA200"),
    ]:
        k = f"EMA{sp}"
        if k in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[k],
                    line=dict(color=col, width=1.5),
                    name=nm,
                    opacity=0.85,
                ),
                row=1,
                col=1,
            )

    if "AVWAP" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["AVWAP"],
                line=dict(color="#a371f7", width=2, dash="dash"),
                name="AVWAP",
            ),
            row=1,
            col=1,
        )

    for cn, cl, nm in [
        ("VWAP_U1", "rgba(163, 113, 247, .35)", "VWAP+1SD"),
        ("VWAP_L1", "rgba(163, 113, 247, .35)", "VWAP-1SD"),
        ("VWAP_U2", "rgba(163, 113, 247, .15)", "VWAP+2SD"),
        ("VWAP_L2", "rgba(163, 113, 247, .15)", "VWAP-2SD"),
        ("VWAP_TOP", "#f85149", "VWAP Top"),
        ("VWAP_BOT", "#3fb950", "VWAP Bot"),
        ("VWAP_RTOP", "rgba(248, 81, 73, .5)", "VWAP R. Top"),
        ("VWAP_RBOT", "rgba(63, 185, 80, .5)", "VWAP R. Bot"),
    ]:
        if cn in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[cn],
                    line=dict(color=cl, width=1.2, dash="dot"),
                    name=nm,
                    opacity=0.75,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

    if "Squeeze_Fire" in df.columns:
        sq = df[df["Squeeze_Fire"]]
        if len(sq):
            fig.add_trace(
                go.Scatter(
                    x=sq.index,
                    y=sq["Low"] * 0.99,
                    mode="markers",
                    marker=dict(symbol="star", color="#a371f7", size=14),
                    name="Squeeze Fire",
                ),
                row=1,
                col=1,
            )

    if "BullPat" in df.columns:
        bp = df[df["BullPat"]]
        if len(bp):
            fig.add_trace(
                go.Scatter(
                    x=bp.index,
                    y=bp["Low"] * 0.985,
                    mode="markers",
                    marker=dict(symbol="triangle-up", color="#3fb950", size=12),
                    name="Bull Pattern",
                ),
                row=1,
                col=1,
            )

    for divs, pc, color, nm in [
        (bull_divs, "Low", "#3fb950", "Bull Div"),
        (bear_divs, "High", "#f85149", "Bear Div"),
    ]:
        for i1, i2 in divs[-1:]:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i1], df.index[i2]],
                    y=[df[pc].iloc[i1], df[pc].iloc[i2]],
                    mode="markers+lines",
                    marker=dict(color=color, size=12),
                    line=dict(color=color, width=1.5, dash="dot"),
                    name=nm,
                ),
                row=1,
                col=1,
            )

    vc = ["#3fb950" if c >= o else "#f85149" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            marker_color=vc,
            opacity=0.55,
            name="Volume",
        ),
        row=2,
        col=1,
    )

    if "Vol_MA20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Vol_MA20"],
                line=dict(color="#e3b341", width=1.2),
                name="Vol MA20",
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            line=dict(color="#58a6ff", width=1.5),
            name="RSI",
        ),
        row=3,
        col=1,
    )

    for lvl, cl in [
        (70, "rgba(248, 81, 73, .4)"),
        (50, "rgba(139, 148, 158, .3)"),
        (30, "rgba(63, 185, 80, .4)"),
    ]:
        fig.add_hline(
            y=lvl,
            line=dict(color=cl, dash="dot", width=1),
            row=3,
            col=1,
        )

    for divs, pc, color in [
        (bull_divs, "Low", "#3fb950"),
        (bear_divs, "High", "#f85149"),
    ]:
        for i1, i2 in divs[-1:]:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[i1], df.index[i2]],
                    y=[df["RSI"].iloc[i1], df["RSI"].iloc[i2]],
                    mode="markers+lines",
                    marker=dict(color=color, size=10),
                    line=dict(color=color, width=1.2, dash="dot"),
                    showlegend=False,
                ),
                row=3,
                col=1,
            )

    mhc = ["#3fb950" if v >= 0 else "#f85149" for v in df["MACD_Hist"]]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["MACD_Hist"],
            marker_color=mhc,
            name="MACD Hist",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD"],
            line=dict(color="#58a6ff", width=1.5),
            name="MACD",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD_Signal"],
            line=dict(color="#f0883e", width=1.2),
            name="Signal",
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        height=900,
        template="plotly_dark",
        paper_bgcolor="#07090f",
        plot_bgcolor="#0d1117",
        font=dict(family="JetBrains Mono", color="#8b949e", size=11),
        legend=dict(
            bgcolor="rgba(13, 17, 23, .85)",
            bordercolor="#21262d",
            borderwidth=1,
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=8, r=8, t=36, b=8),
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor="#21262d", row=i, col=1)
        fig.update_yaxes(gridcolor="#21262d", side="right", row=i, col=1)
    return fig


def plot_equity(eq_list):
    ea = np.array(eq_list)
    pk = np.maximum.accumulate(ea)
    dd = (pk - ea) / pk * 100
    x = list(range(len(ea)))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=["Equity Curve", "Drawdown %"],
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=ea,
            fill="tozeroy",
            line=dict(color="#3fb950", width=2),
            fillcolor="rgba(63, 185, 80, .15)",
            name="Equity",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=-dd,
            fill="tozeroy",
            line=dict(color="#f85149", width=2),
            fillcolor="rgba(248, 81, 73, .15)",
            name="DD%",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=380,
        template="plotly_dark",
        paper_bgcolor="#07090f",
        plot_bgcolor="#0d1117",
        font=dict(family="JetBrains Mono", color="#8b949e"),
        margin=dict(l=8, r=8, t=36, b=8),
    )
    return fig


def gc(g):
    return {
        "A+": "#3fb950",
        "A": "#39d353",
        "B+": "#e3b341",
        "B": "#fb923c",
        "C": "#f85149",
    }.get(g, "#8b949e")

# =========================================================
# ADVANCED QUANT / BACKTEST ANALYTICS
# =========================================================
def summarize_trades(dft: pd.DataFrame):
    if dft.empty:
        return {}
    by_reason = dft.groupby("Reason")["PnL"].agg(["count", "sum", "mean"]).reset_index()
    by_side = dft.assign(Side=np.where(dft["PnL"] >= 0, "Win", "Loss")).groupby("Side")[
        "PnL"
    ].agg(["count", "sum", "mean"]).reset_index()
    return {"by_reason": by_reason, "by_side": by_side}


def trade_distribution_chart(dft: pd.DataFrame):
    if dft.empty:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=dft["Ret_Pct"],
            nbinsx=30,
            marker_color="#58a6ff",
            opacity=0.8,
        )
    )
    fig.update_layout(
        title="Trade Return Distribution (%)",
        template="plotly_dark",
        paper_bgcolor="#07090f",
        plot_bgcolor="#0d1117",
        xaxis_title="Return (%)",
        yaxis_title="Count",
        margin=dict(l=8, r=8, t=36, b=8),
    )
    return fig


def monthly_returns_table(dft: pd.DataFrame):
    if dft.empty:
        return pd.DataFrame()
    df = dft.copy()
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    df["Month"] = df["Exit_Date"].dt.to_period("M")
    mr = df.groupby("Month")["Ret_Pct"].sum().to_frame("Monthly_Return_%")
    mr.index = mr.index.astype(str)
    return mr


def optimize_parameters(
    df: pd.DataFrame,
    risk: float,
    trend_s,
    use_trend: bool,
    swing_range: range,
    trail_values: list[float],
):
    results = []
    for bars in swing_range:
        bull_divs, _ = compute_divergences(df, bars=bars)
        if not bull_divs:
            continue
        for trail in trail_values:
            dft, stats = run_backtest(df, bull_divs, risk, trend_s, use_trend, trail=trail)
            if not stats:
                continue
            results.append(
                {
                    "Swing_Bars": bars,
                    "Trail_ATR": trail,
                    "Trades": stats["Trades"],
                    "Sharpe": stats["Sharpe"],
                    "Calmar": stats["Calmar"],
                    "Win_Rate": stats["Win_Rate"],
                    "Total_Return": stats["Total_Return"],
                    "Max_DD": stats["Max_DD"],
                    "Profit_Factor": stats["Profit_Factor"],
                }
            )
    if not results:
        return pd.DataFrame()
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(
        ["Total_Return", "Sharpe", "Calmar"], ascending=[False, False, False]
    )
    return df_res

# =========================================================
# MAIN
# =========================================================
def main():
    get_autotrade_state()

    # Top banner – use wasted top space
    st.markdown(
        """
<div style="
    background: linear-gradient(135deg, #0d1117, #161b22);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 18px 26px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;">
  <div style="display:flex;align-items:center;gap:16px;">
    <div style="
        width: 44px; height: 44px;
        background: linear-gradient(135deg, #2ea043, #1a7f37);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; font-weight: 900; color:#fff;
        box-shadow: 0 0 16px rgba(35, 134, 54, .4);">
      S
    </div>
    <div>
      <div style="
          font-size: 22px; font-weight: 800; color:#f0f6fc;
          letter-spacing:-0.5px; font-family: Plus Jakarta Sans, sans-serif;">
        SNIPER TERMINAL v2
      </div>
      <div style="
          font-size: 10px; color:#8b949e; letter-spacing:2px; margin-top:2px;">
        RSI+MACD DUAL DIVERGENCE · BB SQUEEZE · ORDER FLOW · VWAP BANDS · AUTO TRADE · QUANT BACKTEST
      </div>
    </div>
  </div>
  <div style="text-align:right;font-size:11px;color:#8b949e;">
    <div>Mode: <span style="color:#58a6ff;">BACKTEST & RESEARCH</span></div>
    <div>Universe: NIFTY50 / NIFTY200 / Custom</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    tab_s, tab_b, tab_at, tab_g = st.tabs(
        ["Screener", "Backtest (Quant+)", "Auto Trade", "Guide"]
    )

    # =====================================================
    # SCREENER
    # =====================================================
    with tab_s:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            universe = st.selectbox("Universe", ["NIFTY50", "NIFTY200", "Custom"])
        with c2:
            interval = st.selectbox("Timeframe", ["1d", "1h", "15m", "1wk"])
        with c3:
            swing_bars = st.slider("Swing Bars", 3, 10, 5)
        with c4:
            min_q = st.slider("Min Quality", 0, 100, 50)

        c5, c6, c7 = st.columns(3)
        with c5:
            use_trend = st.toggle("Weekly Trend Filter", value=True)
        with c6:
            fresh_only = st.toggle("Fresh Signals Only", value=True)
        with c7:
            sq_only = st.toggle("Squeeze Signals Only", value=False)

        if universe == "Custom":
            raw = st.text_input("Tickers (comma-separated)", "HAL.NS, IRCTC.NS")
            tickers = [x.strip() for x in raw.split(",") if x.strip()]
        elif universe == "NIFTY50":
            tickers = NIFTY50
        else:
            tickers = NIFTY200

        if st.button("Run Screener"):
            results = []
            prog = st.progress(0, text="Scanning ...")
            for idx, t in enumerate(tickers):
                r = scan_stock(t, interval, use_trend, fresh_only, swing_bars, min_q)
                if r:
                    if sq_only and not r["Squeeze_Fire"] and not r["In_Squeeze"]:
                        pass
                    else:
                        results.append(r)
                prog.progress((idx + 1) / len(tickers), text=f"Scanning {t}")
            prog.empty()

            if results:
                df_r = pd.DataFrame(results).sort_values("Quality", ascending=False)
                st.session_state["sr"] = df_r
                st.success(
                    f"Found {len(df_r)} setups from {len(tickers)} tickers scanned."
                )
            else:
                st.warning("No setups found. Lower Min Quality or disable filters.")

        if "sr" in st.session_state:
            df_r = st.session_state["sr"]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Signals", len(df_r))
            m2.metric("A+ Grade", int((df_r["Grade"] == "A+").sum()))
            m3.metric("Squeeze Fire", int(df_r["Squeeze_Fire"].sum()))
            m4.metric(
                "Avg Quality", f"{round(df_r['Quality'].mean(), 0):.0f}/100"
            )
            m5.metric("Avg R:R", f"{round(df_r['R_R'].mean(), 2):.2f}x")

            disp = [
                "Ticker",
                "Date",
                "Grade",
                "Quality",
                "Entry",
                "SL",
                "TP1",
                "TP2",
                "R_R",
                "RSI",
                "Vol_Ratio",
                "Squeeze_Fire",
                "OFI",
                "Bull_Pat",
            ]
            st.dataframe(df_r[disp], use_container_width=True, height=280)

            sel = st.selectbox("Select ticker for chart", df_r["Ticker"].tolist())
            if sel:
                row = df_r[df_r["Ticker"] == sel].iloc[0]
                df_f = load_data(sel, interval)
                if not df_f.empty:
                    df_f = compute_indicators(df_f)
                    q = signal_quality(df_f, int(row["i2"]))
                    with st.expander("Signal Quality Breakdown", expanded=True):
                        cols = st.columns(5)
                        for ki, (k, v) in enumerate(q["breakdown"].items()):
                            cols[ki % 5].metric(k.replace("_", " "), f"{v}/10")
                    bd, brd = compute_divergences(df_f, bars=swing_bars)
                    with st.expander(f"Chart -- {sel}", expanded=True):
                        st.plotly_chart(
                            plot_chart(df_f, bd, brd, sel),
                            use_container_width=True,
                        )

                    st.markdown("---")
                    st.markdown("**Quick Add to Auto Trade**")
                    qc1, qc2, qc3 = st.columns(3)
                    at_risk = qc1.number_input(
                        "Risk per trade (INR)", value=2000, key="at_risk_screener"
                    )
                    at_qty = qc2.number_input(
                        "Qty override (0=auto)", value=0, key="at_qty_screener"
                    )
                    if qc3.button(
                        "Add Signal to Auto Trade", key=f"add_at_{sel}"
                    ):
                        auto_qty = (
                            at_qty
                            if at_qty > 0
                            else max(
                                1,
                                int(
                                    at_risk
                                    / max(
                                        row["Entry"] - row["SL"],
                                        0.01,
                                    )
                                ),
                            )
                        )
                        paper_place_order(
                            sel,
                            "BUY",
                            auto_qty,
                            row["Entry"],
                            row["SL"],
                            row["TP1"],
                            "Screener Signal",
                        )
                        st.success(f"Added {sel} to Auto Trade (paper).")

    # =====================================================
    # BACKTEST (QUANT+)
    # =====================================================
    with tab_b:
        top1, top2, top3 = st.columns([2, 1.2, 1.2])
        with top1:
            st.subheader("Single Backtest")
        with top2:
            st.subheader("Parameter Sweep")
        with top3:
            st.subheader("Analytics View")

        st.markdown("---")

        col_left, col_right = st.columns([1.4, 1.6])

        # ---------- LEFT: CONFIG + RUN ----------
        with col_left:
            with st.form("backtest_form"):
                bc1, bc2, bc3, bc4 = st.columns(4)
                bt_tick = bc1.text_input("Ticker", "HAL.NS")
                bt_tf = bc2.selectbox("Timeframe", ["1d", "1h", "15m", "1wk"])
                bt_risk = bc3.number_input(
                    "Risk per Trade (INR)", value=2000, step=500
                )
                bt_years = bc4.slider("Years of Data", 1, 5, 2)

                bc5, bc6, bc7 = st.columns(3)
                bt_trend = bc5.toggle("Weekly Trend Filter", value=True)
                bt_trail = bc6.slider(
                    "Trailing Stop (ATR x)", 1.0, 3.0, 2.0, 0.25
                )
                bt_bars = bc7.slider("Swing Bars", 3, 10, 5)

                adv = st.expander("Advanced Quant Settings", expanded=False)
                with adv:
                    bt_min_trades = st.number_input(
                        "Min trades to consider valid",
                        value=10,
                        min_value=0,
                        step=1,
                    )
                    bt_show_monthly = st.checkbox(
                        "Show Monthly Returns Table", value=True
                    )
                    bt_show_dist = st.checkbox(
                        "Show Trade Distribution", value=True
                    )

                submitted = st.form_submit_button("Run Backtest")

            if submitted:
                with st.spinner("Running backtest ..."):
                    df_bt = load_data(bt_tick, bt_tf, years=bt_years)
                    if df_bt.empty:
                        st.error("No data. Check ticker.")
                    else:
                        df_bt = compute_indicators(df_bt)
                        bull_d, bear_d = compute_divergences(df_bt, bars=bt_bars)
                        ts = get_weekly_trend(bt_tick) if bt_trend else None
                        dft, stats = run_backtest(
                            df_bt, bull_d, bt_risk, ts, bt_trend, bt_trail
                        )
                        if dft.empty or stats.get("Trades", 0) < bt_min_trades:
                            st.info(
                                "No trades generated or below minimum trades. "
                                "Try more data or relax filters."
                            )
                        else:
                            st.session_state["bt_stats"] = stats
                            st.session_state["bt_trades"] = dft
                            st.session_state["bt_df"] = df_bt
                            st.session_state["bt_bull"] = bull_d
                            st.session_state["bt_bear"] = bear_d
                            st.session_state["bt_cfg"] = {
                                "ticker": bt_tick,
                                "tf": bt_tf,
                                "years": bt_years,
                                "risk": bt_risk,
                                "trend": bt_trend,
                                "trail": bt_trail,
                                "bars": bt_bars,
                            }

        # ---------- RIGHT: OPTIMIZATION ----------
        with col_right:
            st.markdown("**Quant Optimization Module**")
            if "bt_df" not in st.session_state:
                st.info(
                    "Run a single backtest first to enable optimization on the same dataset."
                )
            else:
                opt_df = st.session_state["bt_df"]
                opt_cfg = st.session_state["bt_cfg"]
                opt_ts = get_weekly_trend(opt_cfg["ticker"]) if opt_cfg["trend"] else None

                oc1, oc2 = st.columns(2)
                with oc1:
                    swing_min = st.slider("Swing Bars (min)", 3, 10, opt_cfg["bars"])
                    swing_max = st.slider("Swing Bars (max)", swing_min, 15, max(10, swing_min + 2))
                with oc2:
                    trail_min = st.slider("Trail ATR (min)", 1.0, 3.0, opt_cfg["trail"])
                    trail_max = st.slider("Trail ATR (max)", trail_min, 4.0, max(2.5, trail_min + 0.5))
                    trail_step = st.selectbox("Trail ATR step", [0.25, 0.5, 1.0], index=0)

                if st.button("Run Optimization Grid Search"):
                    with st.spinner("Running parameter sweep ..."):
                        swing_range = range(swing_min, swing_max + 1)
                        trail_values = list(
                            np.round(
                                np.arange(trail_min, trail_max + 1e-9, trail_step),
                                2,
                            )
                        )
                        df_opt = optimize_parameters(
                            opt_df,
                            opt_cfg["risk"],
                            opt_ts,
                            opt_cfg["trend"],
                            swing_range,
                            trail_values,
                        )
                        if df_opt.empty:
                            st.warning("No valid parameter combinations produced trades.")
                        else:
                            st.session_state["bt_opt"] = df_opt
                            st.success(
                                f"Optimization complete. {len(df_opt)} combinations evaluated."
                            )

                if "bt_opt" in st.session_state:
                    df_opt = st.session_state["bt_opt"]
                    st.markdown("**Top Parameter Sets (sorted by Total Return / Sharpe / Calmar)**")
                    st.dataframe(df_opt.head(15), use_container_width=True, height=260)

        st.markdown("---")

        # ---------- GLOBAL BACKTEST OUTPUT ----------
        if "bt_stats" in st.session_state:
            stats = st.session_state["bt_stats"]
            dft = st.session_state["bt_trades"]
            eq = stats["Equity"]

            st.markdown("### Performance Snapshot")
            r1 = st.columns(4)
            r1[0].metric(
                "Sharpe",
                stats["Sharpe"],
                delta="Good" if stats["Sharpe"] > 1 else "Low",
            )
            r1[1].metric("Sortino", stats["Sortino"])
            r1[2].metric("Calmar", stats["Calmar"])
            r1[3].metric("Win Rate", f"{stats['Win_Rate']}%")

            r2 = st.columns(4)
            r2[0].metric("Profit Factor", stats["Profit_Factor"])
            r2[1].metric("Max Drawdown", f"-{stats['Max_DD']}%")
            r2[2].metric("Total PnL", f"INR {round(stats['Total_PnL'], 0)}")
            r2[3].metric("Trades", stats["Trades"])

            r3 = st.columns(4)
            r3[0].metric("Avg Win", f"+{stats['Avg_Win']}%")
            r3[1].metric("Avg Loss", f"-{stats['Avg_Loss']}%")
            r3[2].metric("Total Return", f"{stats['Total_Return']}%")
            r3[3].metric("Kelly (theoretical)", f"{stats['Kelly']}%")

            c_eq, c_tr = st.columns([1.4, 1.6])
            with c_eq:
                st.plotly_chart(plot_equity(eq), use_container_width=True)
            with c_tr:
                st.markdown("**Trades Table**")
                st.dataframe(
                    dft[
                        [
                            "Entry_Date",
                            "Exit_Date",
                            "Entry",
                            "Exit",
                            "Qty",
                            "PnL",
                            "Ret_Pct",
                            "Reason",
                        ]
                    ],
                    use_container_width=True,
                    height=320,
                )

            qsum = summarize_trades(dft)
            with st.expander("Advanced Trade Analytics", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    if not qsum:
                        st.write("No trades.")
                    else:
                        st.markdown("**PnL by Exit Reason**")
                        st.dataframe(qsum["by_reason"], use_container_width=True)
                with c2:
                    if not qsum:
                        pass
                    else:
                        st.markdown("**PnL by Side (Win/Loss)**")
                        st.dataframe(qsum["by_side"], use_container_width=True)

                if st.checkbox("Show Trade Return Distribution", value=True):
                    fig_dist = trade_distribution_chart(dft)
                    if fig_dist is not None:
                        st.plotly_chart(fig_dist, use_container_width=True)

                if st.checkbox("Show Monthly Returns Heatmap/Table", value=True):
                    mr = monthly_returns_table(dft)
                    if not mr.empty:
                        st.dataframe(mr, use_container_width=True)

    # =====================================================
    # AUTO TRADE
    # =====================================================
    with tab_at:
        st.subheader("Auto Trade (Paper + Shoonya Stub)")
        c1, c2 = st.columns(2)
        with c1:
            st.toggle(
                "Activate Auto Trade (Paper)",
                value=st.session_state["at_active"],
                key="at_active",
            )
            risk_at = st.number_input(
                "Risk per trade (INR)", value=2000, step=500, key="at_risk_auto"
            )
            if st.button("Check Exits (Paper)"):
                paper_check_exits(risk_at)
                st.success("Checked exits on open paper positions.")
        with c2:
            st.metric("Paper PnL (INR)", round(st.session_state["at_pnl"], 2))
            st.markdown("**Open Positions (Paper)**")
            if st.session_state["at_pos"]:
                st.json(st.session_state["at_pos"])
            else:
                st.write("No open positions.")

        st.markdown("---")
        st.markdown("**Activity Log**")
        if st.session_state["at_log"]:
            for line in reversed(st.session_state["at_log"][-100:]):
                st.write(line)
        else:
            st.write("No activity yet.")

    # =====================================================
    # GUIDE
    # =====================================================
    with tab_g:
        st.subheader("Guide / How to Use")
        st.markdown(
            """
- **Screener**: Scans NIFTY50 / NIFTY200 / custom tickers for RSI+MACD dual divergences, BB squeeze, OFI, and patterns.
- **Backtest (Quant+)**:
  - Run a single backtest with your chosen ticker, timeframe, years, swing bars, and ATR trail.
  - See full performance stats, equity curve, drawdown, trade table, and advanced analytics.
  - Use the **Quant Optimization Module** to sweep over swing bars and ATR trail values and find robust parameter sets.
- **Auto Trade**:
  - Paper engine that simulates entries and exits based on your own signals.
  - Shoonya stub is ready to be wired to your live API when you decide.
"""
        )


if __name__ == "__main__":
    main()
