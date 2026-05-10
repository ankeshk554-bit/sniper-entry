# ============================================================
# SNIPER TERMINAL v2.0 - ANKESH
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
from typing import Optional
import time
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# UNIVERSE
# ============================================================

NIFTY200 = [
"RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS",
"HINDUNILVR.NS","ITC.NS","LT.NS","SBIN.NS","BHARTIARTL.NS",
"KOTAKBANK.NS","HCLTECH.NS","ASIANPAINT.NS","MARUTI.NS","AXISBANK.NS",
"SUNPHARMA.NS","BAJFINANCE.NS","ULTRACEMCO.NS","WIPRO.NS","DMART.NS",
"ADANIENT.NS","ADANIPORTS.NS","TITAN.NS","ONGC.NS","POWERGRID.NS",
"NTPC.NS","JSWSTEEL.NS","TATASTEEL.NS","M&M.NS","BAJAJFINSV.NS",
"HDFCLIFE.NS","SBILIFE.NS","DIVISLAB.NS","DRREDDY.NS","BRITANNIA.NS",
"NESTLEIND.NS","HEROMOTOCO.NS","EICHERMOT.NS","BAJAJ-AUTO.NS",
"COALINDIA.NS","GRASIM.NS","TECHM.NS","CIPLA.NS","SHREECEM.NS",
"BPCL.NS","IOC.NS","HINDALCO.NS","VEDL.NS","UPL.NS","ABB.NS",
"AMBUJACEM.NS","APOLLOHOSP.NS","AUROPHARMA.NS","BANDHANBNK.NS",
"BANKBARODA.NS","BEL.NS","BERGEPAINT.NS","BIOCON.NS","BOSCHLTD.NS",
"CANBK.NS","CHOLAFIN.NS","CUMMINSIND.NS","DABUR.NS","DLF.NS",
"GAIL.NS","GODREJCP.NS","HAVELLS.NS","ICICIPRULI.NS","IGL.NS",
"INDHOTEL.NS","INDIGO.NS","INDUSINDBK.NS","LUPIN.NS","MFSL.NS",
"MUTHOOTFIN.NS","NAUKRI.NS","PIDILITIND.NS","PIIND.NS","PNB.NS",
"POLYCAB.NS","RECLTD.NS","SAIL.NS","SRF.NS","TATACONSUM.NS",
"TATAMOTORS.NS","TATAPOWER.NS","TORNTPHARM.NS","TRENT.NS",
"TVSMOTOR.NS","VOLTAS.NS","ZEEL.NS","HAL.NS","IRCTC.NS",
"DELHIVERY.NS","ZOMATO.NS","PAYTM.NS","NYKAA.NS","LTIM.NS",
]

NIFTY50 = [
"RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS",
"HINDUNILVR.NS","ITC.NS","LT.NS","SBIN.NS","BHARTIARTL.NS",
"KOTAKBANK.NS","HCLTECH.NS","ASIANPAINT.NS","MARUTI.NS","AXISBANK.NS",
"SUNPHARMA.NS","BAJFINANCE.NS","ULTRACEMCO.NS","WIPRO.NS","TITAN.NS",
"ONGC.NS","POWERGRID.NS","NTPC.NS","JSWSTEEL.NS","TATASTEEL.NS",
"M&M.NS","BAJAJFINSV.NS","HDFCLIFE.NS","SBILIFE.NS","DIVISLAB.NS",
"DRREDDY.NS","BRITANNIA.NS","NESTLEIND.NS","HEROMOTOCO.NS",
"EICHERMOT.NS","BAJAJ-AUTO.NS","COALINDIA.NS","GRASIM.NS",
"TECHM.NS","CIPLA.NS","BPCL.NS","IOC.NS","HINDALCO.NS",
"TATACONSUM.NS","TATAMOTORS.NS","TATAPOWER.NS","INDUSINDBK.NS",
"ADANIENT.NS","ADANIPORTS.NS","DMART.NS",
]

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Sniper Terminal v2 - Ankesh",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap');

body, .stApp { background: #07090f !important; color:#c9d1d9 !important; }
.stApp { font-family: 'JetBrains Mono', monospace !important; }

#MainMenu, footer, header { visibility: hidden; }

.sniper-header {
    background: linear-gradient(135deg, #0d1117, #161b22);
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px 24px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 16px;
}

.metric-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}

.quality-bar {
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, #238636, #2ea043);
    margin-top: 4px;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0d1117;
    border-bottom: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    color: #8b949e;
    font-size: 12px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
}

.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: white !important;
    border: none !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border-radius: 6px !important;
    padding: 9px 24px !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    filter: brightness(1.15) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(35,134,54,.4) !important;
}

.stTextInput > div > div > input, .stNumberInput > div > div > input {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stSelectbox > div > div { background:#0d1117 !important; border: 1px solid #30363d !important; }
.stDataFrame { border: 1px solid #21262d !important; border-radius: 8px !important; }

[data-testid="metric-container"] { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; }
[data-testid="stMetricValue"] { font-family: "JetBrains Mono", monospace; font-size: 20px; }

.sig-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
}
.sig-bull { background: rgba(35,134,54,.2); color: #3fb950; border: 1px solid rgba(35,134,54,.4); }
.sig-bear { background: rgba(218,54,51,.2); color: #f85149; border: 1px solid rgba(218,54,51,.4); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LAYER
# ============================================================

@st.cache_data(show_spinner=False, ttl=300)
def load_data(ticker: str, interval: str, years: int = 1) -> pd.DataFrame:
    period_map = {"1wk": "7y", "1d": str(years) + "y", "1h": str(min(years, 2)) + "y", "15m": "60d"}
    df = yf.download(
        ticker,
        period=period_map.get(interval, str(years) + "y"),
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns: return pd.DataFrame()
    return df

# ============================================================
# INDICATOR ENGINE
# ============================================================

@st.cache_data(show_spinner=False)
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, hi, lo, v = df["Close"], df["High"], df["Low"], df["Volume"]
    
    for p in [9, 21, 50, 200]:
        df["EMA" + str(p)] = c.ewm(span=p, adjust=False).mean()
    
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = e12 - e26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=14, adjust=False).mean()

    sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
    df["BB_U"] = sma20 + 2 * std20; df["BB_M"] = sma20; df["BB_L"] = sma20 - 2 * std20
    df["BB_W"] = (df["BB_U"] - df["BB_L"]) / df["BB_M"].replace(0, np.nan)
    
    ema20 = c.ewm(span=20, adjust=False).mean()
    df["KC_U"] = ema20 + 1.5 * df["ATR"]; df["KC_L"] = ema20 - 1.5 * df["ATR"]
    df["Squeeze"] = (df["BB_U"] < df["KC_U"]) & (df["BB_L"] > df["KC_L"])
    df["Squeeze_Fire"] = df["Squeeze"].shift(1).fillna(False) & ~df["Squeeze"]

    lo14 = lo.rolling(14).min(); hi14 = hi.rolling(14).max()
    df["Stoch_K"] = 100 * (c - lo14) / (hi14 - lo14).replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    tp = (hi + lo + c) / 3
    df["AVWAP"] = (tp * v).cumsum() / v.cumsum()
    var = ((tp - df["AVWAP"])**2 * v).cumsum() / v.cumsum()
    sd = np.sqrt(var.clip(lower=0))
    df["VWAP_U1"] = df["AVWAP"] + sd; df["VWAP_L1"] = df["AVWAP"] - sd
    df["VWAP_U2"] = df["AVWAP"] + 2 * sd; df["VWAP_L2"] = df["AVWAP"] - 2 * sd

    df["Vol_MA20"] = v.rolling(20).mean()
    df["Vol_Ratio"] = v / df["Vol_MA20"].replace(0, np.nan)

    body = (c - df["Open"]).abs(); rng = (hi - lo).replace(0, np.nan)
    br = (body / rng).clip(0, 1); bull_b = (c > df["Open"]).astype(float)
    df["OFI"] = (bull_b * br - (1 - bull_b) * br).rolling(5).mean()

    bdy = (c - df["Open"]).abs(); lwck = df[["Open", "Close"]].min(axis=1) - lo; uwck = hi - df[["Open", "Close"]].max(axis=1)
    df["Hammer"] = (lwck > 2 * bdy) & (uwck < bdy * 0.5) & (c > df["Open"])
    
    prev_bear = c.shift(1) < df["Open"].shift(1); curr_bull = c > df["Open"]
    df["BullEngulf"] = prev_bear & curr_bull & (df["Open"] <= c.shift(1)) & (c >= df["Open"].shift(1))
    df["BearEngulf"] = (c.shift(1) > df["Open"].shift(1)) & (c < df["Open"]) & (df["Open"] > c.shift(1))
    df["BullPat"] = df["Hammer"] | df["BullEngulf"]
    df["BearPat"] = df["BearEngulf"] | ((uwck > 2 * bdy) & (lwck < bdy * 0.5) & (c < df["Open"]))
    return df

@st.cache_data(show_spinner=False)
def compute_avwaps(df):
    df = df.copy(); tp = (df["High"] + df["Low"] + df["Close"]) / 3; vol = df["Volume"]
    def avwap(anc):
        out = pd.Series(np.nan, index=df.index)
        if anc >= len(df): return out
        cv = vol.iloc[anc:].cumsum(); ctv = (tp.iloc[anc:] * vol.iloc[anc:]).cumsum()
        out.iloc[anc:] = (ctv / cv.replace(0, np.nan)).values; return out

    tl = int(df["High"].values.argmax()); bl = int(df["Low"].values.argmin())
    rec = max(0, len(df) - 60)
    rt = rec + int(df["High"].iloc[rec:].values.argmax())
    rb = rec + int(df["Low"].iloc[rec:].values.argmin())
    df["VWAP_TOP"] = avwap(tl); df["VWAP_BOT"] = avwap(bl)
    df["VWAP_RTOP"] = avwap(rt); df["VWAP_RBOT"] = avwap(rb)
    return df

# ============================================================
# SWING DETECTION
# ============================================================

def swing_lows(series, bars=5):
    v = series.values; m = np.zeros(len(v), dtype=bool)
    for i in range(bars, len(v) - bars):
        if v[i] == v[i - bars : i + bars + 1].min(): m[i] = True
    return m

def swing_highs(series, bars=5):
    v = series.values; m = np.zeros(len(v), dtype=bool)
    for i in range(bars, len(v) - bars):
        if v[i] == v[i - bars : i + bars + 1].max(): m[i] = True
    return m

# ============================================================
# DIVERGENCE ENGINE
# ============================================================

@st.cache_data(show_spinner=False)
def compute_divergences(df: pd.DataFrame, bars=5):
    lm = swing_lows(df["Low"], bars); hm = swing_highs(df["High"], bars)
    li = np.where(lm)[0]; hi = np.where(hm)[0]
    bull, bear = [], []
    for j in range(1, len(li)):
        i1, i2 = li[j-1], li[j]
        if (df["Low"].iloc[i2] < df["Low"].iloc[i1] and df["RSI"].iloc[i2] > df["RSI"].iloc[i1] and df["MACD_Hist"].iloc[i2] > df["MACD_Hist"].iloc[i1]):
            bull.append((i1, i2))
    for j in range(1, len(hi)):
        i1, i2 = hi[j-1], hi[j]
        if (df["High"].iloc[i2] > df["High"].iloc[i1] and df["RSI"].iloc[i2] < df["RSI"].iloc[i1] and df["MACD_Hist"].iloc[i2] < df["MACD_Hist"].iloc[i1]):
            bear.append((i1, i2))
    return bull[-3:], bear[-3:]

# ============================================================
# SIGNAL QUALITY
# ============================================================

def signal_quality(df, i2):
    sc = {}
    cl = float(df["Close"].iloc[i2]); atr = float(df["ATR"].iloc[i2])
    rsi = float(df["RSI"].iloc[i2])
    sc["RSI_Zone"] = 10 if 28 < rsi < 48 else 6 if rsi < 55 else 2
    mh = float(df["MACD_Hist"].iloc[i2]); mhp = float(df["MACD_Hist"].iloc[i2-1]) if i2 > 0 else mh
    sc["MACD_Turn"] = 10 if mh > mhp and mh > -atr * .01 else 4
    e21 = float(df["EMA21"].iloc[i2]); e50 = float(df["EMA50"].iloc[i2]); e200 = float(df["EMA200"].iloc[i2])
    sc["EMA_Stack"] = 10 if cl > e21 > e50 > e200 else 6 if cl > e50 > e200 else 2
    sc["AVWAP"] = 10 if cl > float(df["AVWAP"].iloc[i2]) else 3
    sqf = bool(df["Squeeze_Fire"].iloc[i2]); sqn = bool(df["Squeeze"].iloc[i2])
    sc["BB_Squeeze"] = 10 if sqf else 6 if sqn else 1
    vr = float(df["Vol_Ratio"].iloc[i2]) if not np.isnan(df["Vol_Ratio"].iloc[i2]) else 1.0
    sc["Vol_Surge"] = 10 if vr > 2 else 7 if vr > 1.5 else 2
    ofi = float(df["OFI"].iloc[i2]) if not np.isnan(df["OFI"].iloc[i2]) else 0.0
    sc["Order_Flow"] = 10 if ofi > .2 else 5 if ofi > 0 else 1
    sc["Candle_Pat"] = 10 if bool(df["BullPat"].iloc[i2]) else 3
    stk = float(df["Stoch_K"].iloc[i2]) if not np.isnan(df["Stoch_K"].iloc[i2]) else 50
    sc["Stochastic"] = 10 if stk < 25 else 5 if stk < 40 else 1
    sc["ATR_Valid"] = 10 if atr > 0 else 0
    tot = sum(sc.values())
    gr = "A+" if tot >= 85 else "A" if tot >= 70 else "B+" if tot >= 55 else "B" if tot >= 40 else "C"
    return {"total": tot, "grade": gr, "breakdown": sc}

@st.cache_data(show_spinner=False)
def get_weekly_trend(ticker):
    df = load_data(ticker, "1wk", years=5)
    if df.empty: return None
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    d = df["Close"].diff()
    g = d.clip(lower=0).ewm(span=14, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + g / l.replace(0, np.nan))
    return (df["Close"] > df["EMA200"]) & (df["RSI"] > 50)

def scan_stock(ticker, interval, use_trend, fresh_only, bars, min_q):
    try:
        df = load_data(ticker, interval)
        if df.empty or len(df) < 100: return None
        df = compute_indicators(df)
        bull, _ = compute_divergences(df, bars=bars)
        if not bull: return None
        i1, i2 = bull[-1]
        if fresh_only and i2 < len(df) - 4: return None
        if use_trend:
            tw = get_weekly_trend(ticker)
            if tw is None: return None
            if not bool(tw.reindex(df.index, method="ffill").iloc[i2]): return None
        ei = i2 + 1
        if ei >= len(df): return None
        ep = float(df["Open"].iloc[ei]); atr = float(df["ATR"].iloc[i2])
        e200 = float(df["EMA200"].iloc[i2]); avwap = float(df["AVWAP"].iloc[i2])
        if ep < e200 or ep < avwap or atr <= 0: return None
        q = signal_quality(df, i2)
        if q["total"] < min_q: return None
        sl = round(ep - 1.5 * atr, 2); tp1 = round(ep + 2 * atr, 2); tp2 = round(ep + 4 * atr, 2)
        rr = round((tp1 - ep) / (ep - sl), 2) if ep != sl else 0
        vr = float(df["Vol_Ratio"].iloc[i2]) if not np.isnan(df["Vol_Ratio"].iloc[i2]) else 1.0
        ofi = float(df["OFI"].iloc[i2]) if not np.isnan(df["OFI"].iloc[i2]) else 0.0
        return {
            "Ticker": ticker, "Date": str(df.index[i2])[:10], "Entry": ep, "SL": sl, "TP1": tp1, "TP2": tp2, "R_R": rr,
            "Quality": q["total"], "Grade": q["grade"], "RSI": round(float(df["RSI"].iloc[i2]), 1),
            "Vol_Ratio": round(vr, 2), "Squeeze_Fire": bool(df["Squeeze_Fire"].iloc[i2]),
            "In_Squeeze": bool(df["Squeeze"].iloc[i2]), "OFI": round(ofi, 3), "Bull_Pat": bool(df["BullPat"].iloc[i2]),
            "ATR": round(atr, 2), "i1": i1, "i2": i2,
        }
    except Exception: return None

# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(df, bull_divs, risk, trend_s, use_trend, trail=2.0):
    df = df.reset_index(drop=True)
    if use_trend and trend_s is not None:
        try: ts = trend_s.reindex(range(len(df)), method="ffill")
        except: ts = pd.Series(True, index=range(len(df)))
    else: ts = pd.Series(True, index=range(len(df)))
    trades = []; eq = [100.0]
    for i1, i2 in bull_divs:
        if not bool(ts.iloc[i2] if i2 < len(ts) else True): continue
        ei = i2 + 1
        if ei >= len(df) - 1: continue
        ep = float(df["Open"].iloc[ei]); atr = float(df["ATR"].iloc[i2])
        e200 = float(df["EMA200"].iloc[i2]); avwap = float(df["AVWAP"].iloc[i2])
        if ep < e200 or ep < avwap or atr <= 0: continue
        sl0 = ep - 1.5 * atr; tp1 = ep + 2 * atr; tp2 = ep + 4 * atr; rpp = ep - sl0
        if rpp <= 0: continue
        qty = max(int(risk / rpp), 1)
        csl = sl0; pdone = False; xp = None; xi = None; xr = "TIME"; rem = qty
        for j in range(ei + 1, min(ei + 20, len(df))):
            hi = float(df["High"].iloc[j]); lo = float(df["Low"].iloc[j]); cl = float(df["Close"].iloc[j])
            csl = max(csl, cl - trail * float(df["ATR"].iloc[j]))
            if lo <= csl: xp, xi, xr = csl, j, "SL"; break
            if not pdone and hi >= tp1:
                pdone = True; half = qty // 2; pnl1 = (tp1 - ep) * half
                trades.append({"Entry_Date": str(df.index[ei])[:10], "Exit_Date": str(df.index[j])[:10], "Entry": round(ep, 2), "Exit": round(tp1, 2), "Qty": half, "PnL": round(pnl1, 2), "Reason": "TP1", "Ret_Pct": round((tp1 - ep) / ep * 100, 2)})
                eq.append(max(eq[-1] * (1 + pnl1 / (risk * 20)), 0.01)); rem = qty - half
            if hi >= tp2: xp, xi, xr = tp2, j, "TP2"; break
        if xp is None:
            li = min(ei + 19, len(df) - 1); xp = float(df["Close"].iloc[li]); xi = li
        pnl = (xp - ep) * rem; ret = (xp - ep) / ep * 100
        trades.append({"Entry_Date": str(df.index[ei])[:10], "Exit_Date": str(df.index[xi])[:10], "Entry": round(ep, 2), "Exit": round(xp, 2), "Qty": rem, "PnL": round(pnl, 2), "Reason": xr, "Ret_Pct": round(ret, 2)})
        eq.append(max(eq[-1] * (1 + pnl / (risk * 20)), 0.01))
    if not trades: return pd.DataFrame(), {}
    dft = pd.DataFrame(trades); rets = dft["Ret_Pct"].values / 100
    wins = rets[rets > 0]; losses = rets[rets < 0]; std = rets.std() if len(rets) > 1 else 1e-9; neg_std = losses.std() if len(losses) > 1 else 1e-9
    sharpe = round(rets.mean() / std * (252**0.5), 2) if std else 0; sortino = round(rets.mean() / neg_std * (252**0.5), 2) if neg_std else 0
    ea = np.array(eq); pk = np.maximum.accumulate(ea); dd = (pk - ea) / pk * 100; max_dd = float(dd.max()); calmar = round((ea[-1] - ea[0]) / ea[0] * 100 / max(max_dd, 0.01), 2)
    wr = round(len(wins) / len(rets) * 100, 1) if len(rets) else 0; aw = round(wins.mean() * 100, 2) if len(wins) else 0; al = round(abs(losses.mean()) * 100, 2) if len(losses) else 0
    pf = round(wins.sum() / abs(losses.sum()), 2) if len(losses) and losses.sum() != 0 else 99
    p = len(wins) / len(rets) if len(rets) else 0.5; a = abs(losses.mean()) if len(losses) else 1e-9; b = wins.mean() if len(wins) else 1e-9
    kelly = round(max(0, p / a - (1 - p) / b) * 100, 1) if a > 0 else 0
    return dft, {"Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar, "Win_Rate": wr, "Avg_Win": aw, "Avg_Loss": al, "Profit_Factor": pf, "Max_DD": round(max_dd, 2), "Total_PnL": round(dft["PnL"].sum(), 2), "Total_Return": round((ea[-1] - ea[0]) / ea[0] * 100, 2), "Trades": len(dft), "Kelly": kelly, "Equity": eq}

# ============================================================
# AUTO TRADE ENGINE
# ============================================================

def get_autotrade_state():
    if "at_active" not in st.session_state: st.session_state["at_active"] = False
    if "at_log" not in st.session_state: st.session_state["at_log"] = []
    if "at_pos" not in st.session_state: st.session_state["at_pos"] = {}
    if "at_pnl" not in st.session_state: st.session_state["at_pnl"] = 0.0
    if "at_mode" not in st.session_state: st.session_state["at_mode"] = "paper"

def paper_place_order(ticker, direction, qty, price, sl, tpl):
    ts = time.strftime("%H:%M:%S")
    entry = {"time": ts, "ticker": ticker, "dir": direction, "qty": qty, "entry": price, "sl": sl, "tpl": tpl, "status": "OPEN", "pnl": 0.0}
    st.session_state["at_pos"][ticker] = entry
    st.session_state["at_log"].append(f"[{ts}] PAPER {direction} {qty}x {ticker} @ {price:.2f} | SL:{sl:.2f} TP:{tpl:.2f}")

def paper_check_exits(df_live, risk_per_trade):
    to_close = []
    for tk, pos in st.session_state["at_pos"].items():
        if pos["status"] != "OPEN": continue
        try:
            df = load_data(tk, "1d", years=1)
            if df.empty: continue
            last = df.iloc[-1]; hi = float(last["High"]); lo = float(last["Low"])
            if lo <= pos["sl"]:
                pnl = (pos["sl"] - pos["entry"]) * pos["qty"]; st.session_state["at_pnl"] += pnl
                st.session_state["at_log"].append(f"[EXIT-SL] {tk} @ {pos['sl']:.2f} | PnL: INR {pnl:.0f}"); to_close.append(tk)
            elif hi >= pos["tpl"]:
                pnl = (pos["tpl"] - pos["entry"]) * pos["qty"]; st.session_state["at_pnl"] += pnl
                st.session_state["at_log"].append(f"[EXIT-TP] {tk} @ {pos['tpl']:.2f} | PnL: INR {pnl:.0f}"); to_close.append(tk)
        except: pass
    for tk in to_close: st.session_state["at_pos"][tk]["status"] = "CLOSED"

def shoonya_place_order(ticker, direction, qty, price):
    return {"status": "ok", "msg": "Shoonya stub connect api_helper.py"}

# ============================================================
# CHARTS
# ============================================================

def plot_chart(df, bull_divs, bear_divs, ticker):
    df = compute_avwaps(df)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.50, 0.14, 0.18, 0.18], vertical_spacing=0.02, subplot_titles=[ticker, "Volume / OFI", "RSI (14)", "MACD"])
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price", increasing_line_color="#3fb950", decreasing_line_color="#f85149", increasing_fillcolor="#3fb950", decreasing_fillcolor="#f85149"), row=1, col=1)
    for sp, col, nm in [(21, "#e3b341", "EMA21"), (50, "#58a6ff", "EMA50"), (200, "#ff9a3c", "EMA200")]:
        if "EMA" + str(sp) in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df["EMA" + str(sp)], line=dict(color=col, width=1.5), name=nm, opacity=.85), row=1, col=1)
    if "AVWAP" in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df["AVWAP"], line=dict(color="#a371f7", width=2, dash="dash"), name="AVWAP"), row=1, col=1)
    for cn, cl, nm in [("VWAP_TOP", "#f85149", "VWAP Top"), ("VWAP_BOTTOM", "#3fb950", "VWAP Bot")]:
        if cn in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df[cn], line=dict(color=cl, width=1.2, dash="dot"), name=nm, opacity=.75), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=["#3fb950" if c >= o else "#f85149" for c, o in zip(df["Close"], df["Open"])], opacity=.55, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#58a6ff", width=1.5), name="RSI"), row=3, col=1)
    for divs, col, color in [(bull_divs, "Low", "#3fb950"), (bear_divs, "High", "#f85149")]:
        for i1, i2 in divs[-1:]: fig.add_trace(go.Scatter(x=[df.index[i1], df.index[i2]], y=[df[col].iloc[i1], df[col].iloc[i2]], mode="markers+lines", marker=dict(color=color, size=12), line=dict(color=color, width=2.5)), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], marker_color=["#3fb950" if v >= 0 else "#f85149" for v in df["MACD_Hist"]], name="MACD Hist"), row=4, col=1)
    fig.update_layout(height=900, template="plotly_dark", paper_bgcolor="#07090f", plot_bgcolor="#0d1117", xaxis_rangeslider_visible=False, margin=dict(l=8, r=8, t=36, b=8))
    return fig

def plot_equity(eq_list):
    ea = np.array(eq_list); pk = np.maximum.accumulate(ea); dd = (pk - ea) / pk * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.65, .35], subplot_titles=["Equity Curve", "Drawdown %"])
    fig.add_trace(go.Scatter(x=list(range(len(ea))), y=ea, fill="tozeroy", line=dict(color="#3fb950", width=2), name="Equity"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(ea))), y=-dd, fill="tozeroy", line=dict(color="#f85149", width=1.5), name="DD%"), row=2, col=1)
    fig.update_layout(height=380, template="plotly_dark", paper_bgcolor="#07090f", plot_bgcolor="#0d1117", margin=dict(l=8, r=8, t=36, b=8))
    return fig

# ============================================================
# MAIN
# ============================================================

def main():
    get_autotrade_state()
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0d1117,#161b22); border-radius:12px; padding:18px 26px; margin-bottom:20px; display: flex; align-items:center; gap: 16px; border:1px solid #21262d;">
    <div style="width: 44px; height: 44px; background: linear-gradient(135deg, #2ea043,#1a7f37); border-radius: 12px; display: flex; align-items:center; justify-content:center; font-size:22px; font-weight:900; color:#fff; box-shadow:0 0 16px rgba(35,134,54,.4)">S</div>
    <div><div style="font-size:22px; font-weight:800; color:#f0f6fc; letter-spacing:-0.5px;">SNIPER TERMINAL v2</div>
    <div style="font-size:10px; color:#8b949e; letter-spacing:2px; margin-top:2px">RSI+MACD DUAL DIVERGENCE BB SQUEEZE | ORDER FLOW | VWAP BANDS AUTO TRADE</div></div></div>
    """, unsafe_allow_html=True)

    tab_s, tab_b, tab_at, tab_g = st.tabs(["Screener", "Backtest", "Auto Trade", "Guide"])
    
    with tab_s:
        c1, c2, c3, c4 = st.columns(4)
        universe = c1.selectbox("Universe", ["NIFTY50", "NIFTY200", "Custom"], index=1)
        interval = c2.selectbox("Timeframe", ["1d", "1h", "15m", "1wk"])
        swing_bars = c3.slider("Swing Bars", 3, 10, 5)
        min_q = c4.slider("Min Quality", 0, 100, 50)
        c5, c6, c7 = st.columns(3)
        use_trend = c5.toggle("Weekly Trend Filter", value=True)
        fresh_only = c6.toggle("Fresh Signals Only", value=True)
        sq_only = c7.toggle("Squeeze Signals Only", value=False)
        tickers = NIFTY50 if universe == "NIFTY50" else NIFTY200 if universe == "NIFTY200" else [x.strip() for x in st.text_input("Tickers", "HAL.NS").split(",")]
        if st.button("Run Screener"):
            results = []
            prog = st.progress(0, text="Scanning...")
            for idx, t in enumerate(tickers):
                r = scan_stock(t, interval, use_trend, fresh_only, swing_bars, min_q)
                if r:
                    if sq_only and not r["Squeeze_Fire"] and not r["In_Squeeze"]: continue
                    results.append(r)
                prog.progress((idx + 1) / len(tickers), text="Scanning " + t)
            prog.empty()
            if results:
                st.session_state["sr"] = pd.DataFrame(results).sort_values("Quality", ascending=False)
                st.success(f"Found {len(results)} setups")
        if "sr" in st.session_state:
            st.dataframe(st.session_state["sr"], use_container_width=True)
            sel = st.selectbox("Select ticker for chart", st.session_state["sr"]["Ticker"].tolist())
            if sel:
                df_f = compute_indicators(load_data(sel, interval))
                bd, brd = compute_divergences(df_f, bars=swing_bars)
                st.plotly_chart(plot_chart(df_f, bd, brd, sel), use_container_width=True)

    with tab_b:
        with st.form("backtest_form"):
            c1, c2, c3, c4 = st.columns(4)
            bt_tick = c1.text_input("Ticker", "HAL.NS")
            bt_tf = c2.selectbox("Timeframe", ["1d", "1h", "15m", "1wk"])
            bt_risk = c3.number_input("Risk", 2000)
            bt_years = c4.slider("Years", 1, 5, 2)
            if st.form_submit_button("Run Backtest"):
                df_bt = compute_indicators(load_data(bt_tick, bt_tf, years=bt_years))
                bd, brd = compute_divergences(df_bt, bars=5)
                dft, stats = run_backtest(df_bt, bd, bt_risk, None, False)
                if not dft.empty: st.write(stats); st.plotly_chart(plot_equity(stats["Equity"]), use_container_width=True)

    with tab_at: st.info("Auto Trade Engine - Testing phase.")
    with tab_g: st.markdown("## Guide\nDetailed trading logic for Sniper Terminal.")

if __name__ == "__main__":
    main()
