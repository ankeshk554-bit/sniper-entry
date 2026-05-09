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

body, .stApp { background: #07090f !important; }
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
    padding: 8px 24px !important;
}
.stButton > button:hover {
    filter: brightness(1.15) !important;
    transform: translateY(-1px) !important;
}

.stDataFrame { border: 1px solid #21262d !important; border-radius: 8px !important; }
.stSelectbox > div, .stTextInput > div { background: #0d1117 !important; }

.sig-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
}
.sig-bull { background: rgba(35,134,54,.2); color: #3fb950; border: 1px solid rgba(35,134,54,.4); }
.sig-bear { background: rgba(218,54,51,.2); color: #f85149; border: 1px solid rgba(218,54,51,.4); }
.sig-neutral { background: rgba(139,148,158,.15); color: #8b949e; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LAYER
# ============================================================

@st.cache_data(show_spinner=False, ttl=300)
def load_data(ticker: str, interval: str, years: int = 1) -> pd.DataFrame:
    period_map = {"1wk": "7y", "1d": f"{years}y", "1h": f"{min(years,2)}y", "15m": "60d"}
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
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
    return df

# ============================================================
# INDICATOR ENGINE
# ============================================================

@st.cache_data(show_spinner=False)
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    for p in [9, 21, 50, 200]:
        df[f"EMA{p}"] = close.ewm(span=p, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=14, adjust=False).mean()

    sma20      = close.rolling(20).mean()
    std20      = close.rolling(20).std()
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Mid"]   = sma20
    df["BB_Lower"] = sma20 - 2 * std20
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]

    ema20        = close.ewm(span=20, adjust=False).mean()
    df["KC_Upper"] = ema20 + 1.5 * df["ATR"]
    df["KC_Lower"] = ema20 - 1.5 * df["ATR"]

    df["Squeeze"] = (df["BB_Upper"] < df["KC_Upper"]) & (df["BB_Lower"] > df["KC_Lower"])
    df["Squeeze_Fire"] = df["Squeeze"].shift(1).fillna(False) & ~df["Squeeze"]

    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["Stoch_K"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    tp = (high + low + close) / 3
    df["AVWAP"] = (tp * vol).cumsum() / vol.cumsum()

    df["VWAP_Var"] = ((tp - df["AVWAP"]) ** 2 * vol).cumsum() / vol.cumsum()
    df["VWAP_SD"]  = np.sqrt(df["VWAP_Var"].clip(lower=0))
    df["VWAP_U1"]  = df["AVWAP"] + 1 * df["VWAP_SD"]
    df["VWAP_U2"]  = df["AVWAP"] + 2 * df["VWAP_SD"]
    df["VWAP_L1"]  = df["AVWAP"] - 1 * df["VWAP_SD"]
    df["VWAP_L2"]  = df["AVWAP"] - 2 * df["VWAP_SD"]

    df["Vol_SMA20"]   = vol.rolling(20).mean()
    df["Vol_Ratio"]   = vol / df["Vol_SMA20"]
    body_size = (close - df["Open"]).abs()
    total_range = (high - low).replace(0, np.nan)
    body_ratio = (body_size / total_range).clip(0, 1)
    bull_body = (close > df["Open"]).astype(float)
    df["OFI"] = (bull_body * body_ratio - (1 - bull_body) * body_ratio).rolling(5).mean()

    df["Hammer"]          = _hammer(df)
    df["BullEngulf"]      = _bull_engulf(df)
    df["BearEngulf"]      = _bear_engulf(df)
    df["MorningStar"]     = _morning_star(df)
    df["ShootingStar"]    = _shooting_star(df)
    df["BullishPattern"]  = df["Hammer"] | df["BullEngulf"] | df["MorningStar"]
    df["BearishPattern"]  = df["BearEngulf"] | df["ShootingStar"]

    return df

def _hammer(df):
    body = (df["Close"] - df["Open"]).abs()
    low_wick = df[["Open","Close"]].min(axis=1) - df["Low"]
    up_wick  = df["High"] - df[["Open","Close"]].max(axis=1)
    return (low_wick > 2 * body) & (up_wick < body * 0.5) & (df["Close"] > df["Open"])

def _bull_engulf(df):
    prev_bear = df["Close"].shift(1) < df["Open"].shift(1)
    curr_bull = df["Close"] > df["Open"]
    engulfs   = (df["Open"] <= df["Close"].shift(1)) & (df["Close"] >= df["Open"].shift(1))
    return prev_bear & curr_bull & engulfs

def _bear_engulf(df):
    prev_bull = df["Close"].shift(1) > df["Open"].shift(1)
    curr_bear = df["Close"] < df["Open"]
    engulfs   = (df["Open"] >= df["Close"].shift(1)) & (df["Close"] <= df["Open"].shift(1))
    return prev_bull & curr_bear & engulfs

def _morning_star(df):
    d1_bear = df["Close"].shift(2) < df["Open"].shift(2)
    d2_small = (df["Close"].shift(1) - df["Open"].shift(1)).abs() < df["ATR"].shift(1) * 0.3
    d3_bull = df["Close"] > df["Open"]
    d3_up   = df["Close"] > (df["Open"].shift(2) + df["Close"].shift(2)) / 2
    return d1_bear & d2_small & d3_bull & d3_up

def _shooting_star(df):
    body   = (df["Close"] - df["Open"]).abs()
    up_wick = df["High"] - df[["Open","Close"]].max(axis=1)
    lo_wick = df[["Open","Close"]].min(axis=1) - df["Low"]
    return (up_wick > 2 * body) & (lo_wick < body * 0.5) & (df["Close"] < df["Open"])

# ============================================================
# ANCHORED VWAP
# ============================================================

@st.cache_data(show_spinner=False)
def compute_anchored_vwaps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"]

    def _avwap(anchor_iloc: int) -> pd.Series:
        out = pd.Series(np.nan, index=df.index)
        if anchor_iloc >= len(df): return out
        sub_tp, sub_vol = tp.iloc[anchor_iloc:], vol.iloc[anchor_iloc:]
        cum_vol = sub_vol.cumsum()
        cum_tpv = (sub_tp * sub_vol).cumsum()
        out.iloc[anchor_iloc:] = (cum_tpv / cum_vol.replace(0, np.nan)).values
        return out

    top_loc, bottom_loc = df["High"].values.argmax(), df["Low"].values.argmin()
    recent = max(0, len(df) - 60)
    recent_top = recent + df["High"].iloc[recent:].values.argmax()
    recent_bot = recent + df["Low"].iloc[recent:].values.argmin()

    df["VWAP_TOP"]        = _avwap(top_loc)
    df["VWAP_BOTTOM"]     = _avwap(bottom_loc)
    df["VWAP_RECENT_TOP"] = _avwap(recent_top)
    df["VWAP_RECENT_BOT"] = _avwap(recent_bot)
    return df

# ============================================================
# SWING DETECTION
# ============================================================

def _swing_lows(series: pd.Series, bars: int = 5) -> np.ndarray:
    vals = series.values
    mask = np.zeros(len(vals), dtype=bool)
    for i in range(bars, len(vals) - bars):
        window = vals[i - bars: i + bars + 1]
        if vals[i] == window.min(): mask[i] = True
    return mask

def _swing_highs(series: pd.Series, bars: int = 5) -> np.ndarray:
    vals = series.values
    mask = np.zeros(len(vals), dtype=bool)
    for i in range(bars, len(vals) - bars):
        window = vals[i - bars: i + bars + 1]
        if vals[i] == window.max(): mask[i] = True
    return mask

# ============================================================
# DIVERGENCE ENGINE
# ============================================================

@st.cache_data(show_spinner=False)
def compute_divergences(df: pd.DataFrame, swing_bars: int = 5):
    low_mask  = _swing_lows(df["Low"],  bars=swing_bars)
    high_mask = _swing_highs(df["High"], bars=swing_bars)
    low_idxs  = np.where(low_mask)[0]
    high_idxs = np.where(high_mask)[0]
    bull_divs, bear_divs = [], []

    for j in range(1, len(low_idxs)):
        i1, i2 = low_idxs[j - 1], low_idxs[j]
        if df["Low"].iloc[i2] < df["Low"].iloc[i1] and df["RSI"].iloc[i2] > df["RSI"].iloc[i1] and df["MACD_Hist"].iloc[i2] > df["MACD_Hist"].iloc[i1]:
            bull_divs.append((i1, i2))

    for j in range(1, len(high_idxs)):
        i1, i2 = high_idxs[j - 1], high_idxs[j]
        if df["High"].iloc[i2] > df["High"].iloc[i1] and df["RSI"].iloc[i2] < df["RSI"].iloc[i1] and df["MACD_Hist"].iloc[i2] < df["MACD_Hist"].iloc[i1]:
            bear_divs.append((i1, i2))

    return bull_divs[-3:], bear_divs[-3:]

# ============================================================
# SIGNAL QUALITY SCORE
# ============================================================

def compute_signal_quality(df: pd.DataFrame, i2: int) -> dict:
    scores = {}
    close, atr = float(df["Close"].iloc[i2]), float(df["ATR"].iloc[i2])
    rsi = float(df["RSI"].iloc[i2])
    scores["RSI_Zone"]    = 10 if 28 < rsi < 48 else 6 if rsi < 55 else 2
    mh_now  = float(df["MACD_Hist"].iloc[i2])
    mh_prev = float(df["MACD_Hist"].iloc[i2 - 1]) if i2 > 0 else mh_now
    scores["MACD_Turn"]   = 10 if mh_now > mh_prev and mh_now > -atr * 0.01 else 4
    e21, e50, e200 = float(df["EMA21"].iloc[i2]), float(df["EMA50"].iloc[i2]), float(df["EMA200"].iloc[i2])
    scores["EMA_Stack"]   = 10 if close > e21 > e50 > e200 else 6 if close > e50 > e200 else 2
    avwap = float(df["AVWAP"].iloc[i2])
    scores["AVWAP"]       = 10 if close > avwap else 3
    sq_fire = bool(df["Squeeze_Fire"].iloc[i2]) if "Squeeze_Fire" in df.columns else False
    sq_now  = bool(df["Squeeze"].iloc[i2]) if "Squeeze" in df.columns else False
    scores["BB_Squeeze"]  = 10 if sq_fire else 6 if sq_now else 1
    vol_r = float(df["Vol_Ratio"].iloc[i2]) if "Vol_Ratio" in df.columns else 1.0
    scores["Volume_Surge"]= 10 if vol_r > 2.0 else 7 if vol_r > 1.5 else 2
    ofi = float(df["OFI"].iloc[i2]) if "OFI" in df.columns else 0
    scores["Order_Flow"]  = 10 if ofi > 0.2 else 5 if ofi > 0 else 1
    bull_pat = bool(df["BullishPattern"].iloc[i2]) if "BullishPattern" in df.columns else False
    scores["Candle_Pattern"] = 10 if bull_pat else 3
    stoch = float(df["Stoch_K"].iloc[i2]) if "Stoch_K" in df.columns else 50
    scores["Stochastic"]  = 10 if stoch < 25 else 5 if stoch < 40 else 1
    scores["ATR_Valid"]   = 10 if atr > 0 else 0
    total = sum(scores.values())
    grade = "A+" if total >= 85 else "A" if total >= 70 else "B+" if total >= 55 else "B" if total >= 40 else "C"
    return {"total": total, "grade": grade, "breakdown": scores}

# ============================================================
# WEEKLY TREND FILTER
# ============================================================

@st.cache_data(show_spinner=False)
def get_weekly_trend(ticker: str) -> Optional[pd.Series]:
    df_w = load_data(ticker, "1wk", years=5)
    if df_w.empty: return None
    df_w["EMA200"] = df_w["Close"].ewm(span=200, adjust=False).mean()
    delta = df_w["Close"].diff()
    gain, loss = delta.clip(lower=0).ewm(span=14, adjust=False).mean(), (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df_w["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df_w["TrendW"] = (df_w["Close"] > df_w["EMA200"]) & (df_w["RSI"] > 50)
    return df_w["TrendW"]

# ============================================================
# SCREENER ENGINE
# ============================================================

def scan_stock(ticker: str, interval: str, use_trend: bool, fresh_only: bool, swing_bars: int, min_quality: int) -> Optional[dict]:
    try:
        df = load_data(ticker, interval)
        if df.empty or len(df) < 100: return None
        df = compute_indicators(df)
        bull_divs, _ = compute_divergences(df, swing_bars=swing_bars)
        if not bull_divs: return None
        i1, i2 = bull_divs[-1]
        if fresh_only and i2 < len(df) - 4: return None
        if use_trend:
            tw = get_weekly_trend(ticker)
            if tw is None or not bool(tw.reindex(df.index, method="ffill").iloc[i2]): return None
        entry_idx = i2 + 1
        if entry_idx >= len(df): return None
        entry_price, atr_val, ema200_val, avwap_val = float(df["Open"].iloc[entry_idx]), float(df["ATR"].iloc[i2]), float(df["EMA200"].iloc[i2]), float(df["AVWAP"].iloc[i2])
        if entry_price < ema200_val or entry_price < avwap_val or atr_val <= 0: return None
        quality = compute_signal_quality(df, i2)
        if quality["total"] < min_quality: return None
        sl, tp1, tp2 = round(entry_price - 1.5 * atr_val, 2), round(entry_price + 2.0 * atr_val, 2), round(entry_price + 4.0 * atr_val, 2)
        rr = round((tp1 - entry_price) / (entry_price - sl), 2)
        return {
            "Ticker": ticker, "Signal_Date": df.index[i2].strftime("%Y-%m-%d"), "Entry": entry_price, "SL": sl, "TP1": tp1, "TP2": tp2, "R:R": rr,
            "Quality": quality["total"], "Grade": quality["grade"], "RSI": round(float(df["RSI"].iloc[i2]), 1),
            "Vol_Ratio": round(float(df["Vol_Ratio"].iloc[i2]), 2), "Squeeze_Fire": bool(df["Squeeze_Fire"].iloc[i2]),
            "In_Squeeze": bool(df["Squeeze"].iloc[i2]), "OFI": round(float(df["OFI"].iloc[i2]), 3),
            "Bull_Pattern": bool(df["BullishPattern"].iloc[i2]), "i1": i1, "i2": i2, "ATR": round(atr_val, 2),
        }
    except Exception: return None

# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(df: pd.DataFrame, bull_divs: list, risk_per_trade: float, trend_series, use_trend: bool, trailing_atr_mult: float = 2.0, swing_bars: int = 5) -> tuple[pd.DataFrame, dict]:
    df = df.reset_index(drop=True)
    ts = trend_series.reindex(df.index, method="ffill") if use_trend and trend_series is not None else pd.Series(True, index=df.index)
    trades, equity_curve = [], [100.0]
    for i1, i2 in bull_divs:
        if use_trend and not bool(ts.iloc[i2] if len(ts) > i2 else True): continue
        entry_idx = i2 + 1
        if entry_idx >= len(df) - 1: continue
        entry_p, atr_v, ema200_v, avwap_v = float(df["Open"].iloc[entry_idx]), float(df["ATR"].iloc[i2]), float(df["EMA200"].iloc[i2]), float(df["AVWAP"].iloc[i2])
        if entry_p < ema200_v or entry_p < avwap_v or atr_v <= 0: continue
        sl_base, tp1, tp2 = entry_p - 1.5 * atr_v, entry_p + 2.0 * atr_v, entry_p + 4.0 * atr_v
        risk_pp = entry_p - sl_base
        if risk_pp <= 0: continue
        qty = max(int(risk_per_trade / risk_pp), 1)
        current_sl, partial_done, exit_p, exit_idx, exit_reason = sl_base, False, None, None, "TIME"
        for j in range(entry_idx + 1, min(entry_idx + 20, len(df))):
            hi, lo, cl = float(df["High"].iloc[j]), float(df["Low"].iloc[j]), float(df["Close"].iloc[j])
            current_sl = max(current_sl, cl - trailing_atr_mult * float(df["ATR"].iloc[j]))
            if lo <= current_sl: exit_p, exit_idx, exit_reason = current_sl, j, "SL"; break
            if not partial_done and hi >= tp1:
                partial_done = True; pnl_p = (tp1 - entry_p) * (qty // 2)
                trades.append({"Entry_Date": df.index[entry_idx], "Exit_Date": df.index[j], "Entry_Price": round(entry_p, 2), "Exit_Price": round(tp1, 2), "Qty": qty // 2, "PnL": round(pnl_p, 2), "Exit_Reason": "TP1", "Risk": round(risk_pp, 2), "Return_Pct": round((tp1 - entry_p) / entry_p * 100, 2)})
                equity_curve.append(equity_curve[-1] * (1 + pnl_p / (risk_per_trade * 20))); qty -= qty // 2
            if hi >= tp2: exit_p, exit_idx, exit_reason = tp2, j, "TP2"; break
        if exit_p is None: exit_p, exit_idx = float(df["Close"].iloc[min(entry_idx+19, len(df)-1)]), min(entry_idx+19, len(df)-1)
        pnl = (exit_p - entry_p) * qty; ret = (exit_p - entry_p) / entry_p * 100
        trades.append({"Entry_Date": df.index[entry_idx], "Exit_Date": df.index[exit_idx], "Entry_Price": round(entry_p, 2), "Exit_Price": round(exit_p, 2), "Qty": qty, "PnL": round(pnl, 2), "Exit_Reason": exit_reason, "Risk": round(risk_pp, 2), "Return_Pct": round(ret, 2)})
        equity_curve.append(max(equity_curve[-1] * (1 + pnl / (risk_per_trade * 20)), 0.01))
    if not trades: return pd.DataFrame(), {}
    df_t = pd.DataFrame(trades); rets = df_t["Return_Pct"].values / 100; wins, losses = rets[rets > 0], rets[rets < 0]
    std_r = rets.std() if len(rets) > 1 else 1e-6
    sharpe = round(rets.mean() / std_r * np.sqrt(252), 2) if std_r else 0
    eq_arr = np.array(equity_curve); peak = np.maximum.accumulate(eq_arr); dd = (peak - eq_arr) / peak; max_dd = dd.max() * 100
    p = len(wins) / len(rets); a, b = abs(losses.mean()) if len(losses) else 1e-6, wins.mean() if len(wins) else 1e-6
    return df_t, {"Total_Trades": len(df_t), "Win_Rate": round(p*100,1), "Avg_Win_Pct": round(b*100,2), "Avg_Loss_Pct": round(a*100,2), "Sharpe": sharpe, "Max_DD_Pct": round(max_dd, 2), "Total_PnL": round(df_t["PnL"].sum(), 2), "Equity_Curve": equity_curve, "Profit_Factor": round(wins.sum()/abs(losses.sum()),2) if len(losses) else 99, "Kelly_Pct": round(max(0, p/a - (1-p)/b)*100,1) if a>0 and b>0 else 0, "Risk_of_Ruin_Pct": round(max(0, (1-(rets.mean()/std_r))**100*100),2) if std_r else 0}

# ============================================================
# PLOTLY CHART
# ============================================================

def plot_chart(df: pd.DataFrame, bull_divs: list, bear_divs: list, ticker: str) -> go.Figure:
    df_p = compute_anchored_vwaps(df)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.52, 0.14, 0.18, 0.16], vertical_spacing=0.02, subplot_titles=[ticker, "Volume + OFI", "RSI (14) + Divergence", "MACD Histogram"])
    fig.add_trace(go.Candlestick(x=df_p.index, open=df_p["Open"], high=df_p["High"], low=df_p["Low"], close=df_p["Close"], name="Price", increasing_line_color="#3fb950", decreasing_line_color="#f85149", increasing_fillcolor="#3fb950", decreasing_fillcolor="#f85149"), row=1, col=1)
    for span, color, name in [(21,"#e3b341","EMA 21"),(50,"#58a6ff","EMA 50"),(200,"#ff9a3c","EMA 200")]:
        if f"EMA{span}" in df_p.columns: fig.add_trace(go.Scatter(x=df_p.index, y=df_p[f"EMA{span}"], line=dict(color=color, width=1.5), name=name, opacity=0.8), row=1, col=1)
    if "AVWAP" in df_p.columns: fig.add_trace(go.Scatter(x=df_p.index, y=df_p["AVWAP"], line=dict(color="#a371f7", width=2, dash="dash"), name="AVWAP"), row=1, col=1)
    for col, color, nm in [("VWAP_TOP","#f85149","VWAP Top"),("VWAP_BOTTOM","#3fb950","VWAP Bot")]:
        if col in df_p.columns: fig.add_trace(go.Scatter(x=df_p.index, y=df_p[col], line=dict(color=color, width=1.2, dash="dot"), name=nm, opacity=0.7), row=1, col=1)
    fig.add_trace(go.Bar(x=df_p.index, y=df_p["Volume"], marker_color=["#3fb950" if c>=o else "#f85149" for c,o in zip(df_p["Close"], df_p["Open"])], opacity=0.6, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_p.index, y=df_p["RSI"], line=dict(color="#58a6ff", width=1.5), name="RSI"), row=3, col=1)
    for divs, col, color in [(bull_divs, "Low", "#3fb950"), (bear_divs, "High", "#f85149")]:
        for i1, i2 in divs[-1:]: fig.add_trace(go.Scatter(x=[df_p.index[i1], df_p.index[i2]], y=[df_p[col].iloc[i1], df_p[col].iloc[i2]], mode="markers+lines", marker=dict(color=color, size=12), line=dict(color=color, width=2.5)), row=1, col=1)
    fig.add_trace(go.Bar(x=df_p.index, y=df_p["MACD_Hist"], marker_color=["#3fb950" if v>=0 else "#f85149" for v in df_p["MACD_Hist"]], name="MACD Hist"), row=4, col=1)
    fig.update_layout(height=900, template="plotly_dark", paper_bgcolor="#07090f", plot_bgcolor="#0d1117", xaxis_rangeslider_visible=False, margin=dict(l=8, r=8, t=36, b=8))
    return fig

def plot_equity(equity_curve: list, df_trades: pd.DataFrame) -> go.Figure:
    eq = np.array(equity_curve); peak = np.maximum.accumulate(eq); dd = (peak - eq) / peak * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], subplot_titles=["Equity Curve", "Drawdown %"])
    fig.add_trace(go.Scatter(x=list(range(len(eq))), y=eq, fill="tozeroy", line=dict(color="#3fb950", width=2), name="Equity"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(eq))), y=-dd, fill="tozeroy", line=dict(color="#f85149", width=1.5), name="Drawdown %"), row=2, col=1)
    fig.update_layout(height=400, template="plotly_dark", paper_bgcolor="#07090f", plot_bgcolor="#0d1117")
    return fig

def grade_color(g): return {"A+":"#3fb950","A":"#39d353","B+":"#e3b341","B":"#fb923c","C":"#8b949e"}.get(g,"#8b949e")

# ============================================================
# MAIN UI
# ============================================================

def main():
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d1117,#161b22);border:1px solid #21262d;border-radius:10px;padding:16px 24px;margin-bottom:20px;display:flex;align-items:center;gap:16px'>
    <div style='width:40px;height:40px;background:linear-gradient(135deg,#2ea043,#1a7f37);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px;color:white'>S</div>
    <div><div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#f0f6fc;letter-spacing:-0.5px'>SNIPER TERMINAL v2</div>
    <div style='font-size:10px;color:#8b949e;letter-spacing:2px;font-family:JetBrains Mono'>RSI+MACD DUAL DIVERGENCE * BB SQUEEZE * ORDER FLOW * VWAP BANDS * WALK-FORWARD BACKTEST</div></div></div>
    """, unsafe_allow_html=True)

    tab_screen, tab_back, tab_guide = st.tabs(["Screener", "Backtest", "Guide"])
    with tab_screen:
        c1, c2, c3, c4 = st.columns(4)
        universe, interval_s, swing_bars, min_quality = c1.selectbox("Universe", ["NIFTY50","NIFTY200","Custom"], index=1), c2.selectbox("Timeframe", ["1d","1h","15m","1wk"]), c3.slider("Swing Lookback", 3, 10, 5), c4.slider("Min Quality", 0, 100, 50)
        use_trend, fresh_only, show_squeeze = st.toggle("Weekly Trend Filter", value=True), st.toggle("Fresh Only", value=True), st.toggle("Squeeze Only", value=False)
        tickers = NIFTY50 if universe=="NIFTY50" else NIFTY200 if universe=="NIFTY200" else [x.strip() for x in st.text_input("Tickers", "HAL.NS").split(",")]
        if st.button("> Run Screener"):
            results = []
            prog = st.progress(0)
            for i, t in enumerate(tickers):
                r = scan_stock(t, interval_s, use_trend, fresh_only, swing_bars, min_quality)
                if r:
                    if show_squeeze and not r["Squeeze_Fire"] and not r["In_Squeeze"]: continue
                    results.append(r)
                prog.progress((i+1)/len(tickers))
            if results: st.session_state["screen_results"] = pd.DataFrame(results).sort_values("Quality", ascending=False)
            else: st.warning("No setups found.")
        if "screen_results" in st.session_state:
            df_r = st.session_state["screen_results"]
            st.dataframe(df_r, use_container_width=True)
            sel = st.selectbox("View Chart", df_r["Ticker"].tolist())
            if sel:
                df_f = compute_indicators(load_data(sel, interval_s))
                b, br = compute_divergences(df_f, swing_bars)
                st.plotly_chart(plot_chart(df_f, b, br, sel), use_container_width=True)

    with tab_back:
        bt_ticker, bt_tf, bt_risk = st.text_input("Ticker", "HAL.NS"), st.selectbox("TF", ["1d","1h","15m"], key="bt_tf"), st.number_input("Risk", 2000)
        if st.button("Run Backtest"):
            df_bt = compute_indicators(load_data(bt_ticker, bt_tf, 2))
            b, br = compute_divergences(df_bt, 5)
            df_t, stats = run_backtest(df_bt, b, bt_risk, None, False)
            if not df_t.empty:
                st.write(stats)
                st.plotly_chart(plot_equity(stats["Equity_Curve"], df_t), use_container_width=True)
                st.dataframe(df_t, use_container_width=True)

    with tab_guide:
        st.info("Sniper Terminal v2.0 - Built for Ankesh. Uses dual RSI/MACD divergence and Keltner squeeze logic.")

if __name__ == "__main__":
    main()
