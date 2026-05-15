# ============================================================
# SNIPER TERMINAL v4 — ROYAL GOLD EDITION (FULL FUSION BUILD)
# ============================================================

import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG & GLOBAL STATE
# ============================================================

st.set_page_config(
    page_title="Sniper Terminal v4 — Royal Gold",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global defaults
st.session_state.setdefault("theme", "Dark")
st.session_state.setdefault("total_capital", 200000.0)
st.session_state.setdefault("risk_pct", 1.0)
st.session_state.setdefault("selected_setup", "Divergence")
st.session_state.setdefault("nav_page", "Screener")

# ============================================================
# ROYAL GOLD THEME ENGINE
# ============================================================

def apply_theme():
    theme = st.session_state.get("theme", "Dark")

    if theme == "Dark":
        bg = "#050608"
        fg = "#f5f5f5"
        card = "#0b0d11"
        border = "#2b2f36"
        gold = "#f7e7ce"
        gold_soft = "#c9b79a"
        plot_theme = "plotly_dark"
    else:
        bg = "#f5f5f7"
        fg = "#111111"
        card = "#ffffff"
        border = "#d0d0d5"
        gold = "#c9a96a"
        gold_soft = "#b09055"
        plot_theme = "plotly_white"

    st.session_state["plot_theme"] = plot_theme
    st.session_state["gold"] = gold

    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {bg};
            --fg: {fg};
            --card: {card};
            --border: {border};
            --gold: {gold};
            --gold-soft: {gold_soft};
        }}

        body, .stApp {{
            background: var(--bg) !important;
            color: var(--fg) !important;
        }}

        .royal-card {{
            background: var(--card);
            border-radius: 12px;
            border: 1px solid var(--border);
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 0 18px rgba(0,0,0,0.35);
        }}

        .royal-title {{
            font-size: 22px;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--gold);
        }}

        .royal-subtitle {{
            font-size: 11px;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: rgba(247,231,206,0.75);
        }}

        [data-testid="stSidebar"] {{
            background: radial-gradient(circle at top, #151821 0, #050608 55%);
            border-right: 1px solid rgba(247,231,206,0.25);
        }}

        .stButton > button {{
            background: linear-gradient(135deg, var(--gold), var(--gold-soft)) !important;
            color: #111 !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            padding: 8px 20px !important;
        }}

        .stButton > button:hover {{
            filter: brightness(1.08) !important;
            box-shadow: 0 0 18px rgba(247,231,206,0.45) !important;
        }}

        .stSelectbox > div > div,
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {{
            background: var(--card) !important;
            color: var(--fg) !important;
            border-radius: 8px !important;
            border: 1px solid var(--border) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# DATA LAYER — MERGED v2 + v3
# ============================================================

@st.cache_data(show_spinner=False, ttl=120)
def load_data(ticker: str, interval: str, years: int = 1) -> pd.DataFrame:
    """
    Unified data loader:
    - Supports all intervals from v2 + v3
    - Auto-adjusts period based on interval
    - Fixes MultiIndex columns
    - Cleans NaNs
    """

    period_map = {
        "1wk": "7y",
        "1d": f"{years}y",
        "1h": f"{min(years, 2)}y",
        "15m": "60d",
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

    # Fix MultiIndex (common in NSE)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    return df
# ============================================================
# CORE INDICATORS — OPTIMIZED FUSION (v2 + v3)
# ============================================================

def compute_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core indicator engine combining:
    - EMA stack
    - RSI
    - MACD + Signal + Histogram
    - ATR
    - Bollinger Bands
    - Keltner Channels
    - Squeeze logic
    - Stochastic
    - Volume ratio
    - Order Flow Imbalance (OFI)
    - Candlestick patterns (Hammer, Engulfing)
    """

    df = df.copy()
    c = df["Close"]
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    # -----------------------------
    # EMA STACK
    # -----------------------------
    for p in [9, 21, 50, 200]:
        df[f"EMA{p}"] = c.ewm(span=p, adjust=False).mean()

    # -----------------------------
    # RSI (14)
    # -----------------------------
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # -----------------------------
    # MACD (12, 26, 9)
    # -----------------------------
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # -----------------------------
    # ATR (14)
    # -----------------------------
    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()],
        axis=1
    ).max(axis=1)
    df["ATR"] = tr.ewm(span=14, adjust=False).mean()

    # -----------------------------
    # BOLLINGER BANDS (20, 2)
    # -----------------------------
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["BB_U"] = sma20 + 2 * std20
    df["BB_M"] = sma20
    df["BB_L"] = sma20 - 2 * std20
    df["BB_W"] = (df["BB_U"] - df["BB_L"]) / df["BB_M"].replace(0, np.nan)

    # -----------------------------
    # KELTNER CHANNELS (20, 1.5 ATR)
    # -----------------------------
    ema20 = c.ewm(span=20, adjust=False).mean()
    df["KC_U"] = ema20 + 1.5 * df["ATR"]
    df["KC_L"] = ema20 - 1.5 * df["ATR"]

    # -----------------------------
    # SQUEEZE LOGIC
    # -----------------------------
    df["Squeeze"] = (df["BB_U"] < df["KC_U"]) & (df["BB_L"] > df["KC_L"])
    df["Squeeze_Fire"] = df["Squeeze"].shift(1).fillna(False) & ~df["Squeeze"]

    # -----------------------------
    # STOCHASTIC (14)
    # -----------------------------
    lo14 = l.rolling(14).min()
    hi14 = h.rolling(14).max()
    df["Stoch_K"] = 100 * (c - lo14) / (hi14 - lo14).replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # -----------------------------
    # VOLUME RATIO (20)
    # -----------------------------
    df["Vol_MA20"] = v.rolling(20).mean()
    df["Vol_Ratio"] = v / df["Vol_MA20"].replace(0, np.nan)

    # -----------------------------
    # ORDER FLOW IMBALANCE (OFI)
    # -----------------------------
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    br = (body / rng).clip(0, 1)
    bull = (c > o).astype(float)
    df["OFI"] = (bull * br - (1 - bull) * br).rolling(5).mean()

    # -----------------------------
    # CANDLESTICK PATTERNS
    # -----------------------------

    # Hammer
    lower_wick = df[["Open", "Close"]].min(axis=1) - l
    upper_wick = h - df[["Open", "Close"]].max(axis=1)
    df["Hammer"] = (lower_wick > 2 * body) & (upper_wick < body * 0.5) & (c > o)

    # Bullish Engulfing
    prev_bear = c.shift(1) < o.shift(1)
    curr_bull = c > o
    df["BullEngulf"] = (
        prev_bear
        & curr_bull
        & (o <= c.shift(1))
        & (c >= o.shift(1))
    )

    # Bearish Engulfing
    df["BearEngulf"] = (
        (c.shift(1) > o.shift(1))
        & (c < o)
        & (o > c.shift(1))
    )

    # Combined bullish/bearish patterns
    df["BullPat"] = df["Hammer"] | df["BullEngulf"]
    df["BearPat"] = df["BearEngulf"] | (
        (upper_wick > 2 * body) & (lower_wick < body * 0.5) & (c < o)
    )

    return df
# ============================================================
# ADVANCED INDICATORS — AVWAP, VWAP ANCHORS, SWINGS, DIVERGENCE
# ============================================================

def compute_avwaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes anchored VWAPs from:
    - All‑time high
    - All‑time low
    - Recent 60‑bar high
    - Recent 60‑bar low
    """

    df = df.copy()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"]

    def avwap(anchor_index: int) -> pd.Series:
        """Anchored VWAP from a specific index."""
        out = pd.Series(np.nan, index=df.index)
        if anchor_index >= len(df):
            return out

        cv = vol.iloc[anchor_index:].cumsum()
        ctv = (tp.iloc[anchor_index:] * vol.iloc[anchor_index:]).cumsum()
        out.iloc[anchor_index:] = (ctv / cv.replace(0, np.nan)).values
        return out

    # Global anchors
    top_idx = int(df["High"].values.argmax())
    bot_idx = int(df["Low"].values.argmin())

    # Recent anchors (last 60 bars)
    rec = max(0, len(df) - 60)
    rtop_idx = rec + int(df["High"].iloc[rec:].values.argmax())
    rbot_idx = rec + int(df["Low"].iloc[rec:].values.argmin())

    df["vwap_top"] = avwap(top_idx)
    df["vwap_bot"] = avwap(bot_idx)
    df["vwap_rtop"] = avwap(rtop_idx)
    df["vwap_rbot"] = avwap(rbot_idx)

    return df


# ============================================================
# SWING HIGH / LOW DETECTORS (Optimized)
# ============================================================

def swing_lows(series: pd.Series, bars=5):
    """
    Finds swing lows using a rolling window.
    """
    v = series.values
    m = np.zeros(len(v), dtype=bool)
    for i in range(bars, len(v) - bars):
        if v[i] == v[i - bars : i + bars + 1].min():
            m[i] = True
    return m


def swing_highs(series: pd.Series, bars=5):
    """
    Finds swing highs using a rolling window.
    """
    v = series.values
    m = np.zeros(len(v), dtype=bool)
    for i in range(bars, len(v) - bars):
        if v[i] == v[i - bars : i + bars + 1].max():
            m[i] = True
    return m


# ============================================================
# DIVERGENCE HELPERS (Bullish + Bearish)
# ============================================================

def compute_divergences(df: pd.DataFrame, bars=5):
    """
    Detects bullish and bearish divergences using:
    - Price swing highs/lows
    - RSI
    - MACD Histogram
    """

    lm = swing_lows(df["Low"], bars)
    hm = swing_highs(df["High"], bars)

    li = np.where(lm)[0]
    hi = np.where(hm)[0]

    bull, bear = [], []

    # Bullish divergence
    for j in range(1, len(li)):
        i1, i2 = li[j - 1], li[j]
        if (
            df["Low"].iloc[i2] < df["Low"].iloc[i1] and
            df["RSI"].iloc[i2] > df["RSI"].iloc[i1] and
            df["MACD_Hist"].iloc[i2] > df["MACD_Hist"].iloc[i1]
        ):
            bull.append((i1, i2))

    # Bearish divergence
    for j in range(1, len(hi)):
        i1, i2 = hi[j - 1], hi[j]
        if (
            df["High"].iloc[i2] > df["High"].iloc[i1] and
            df["RSI"].iloc[i2] < df["RSI"].iloc[i1] and
            df["MACD_Hist"].iloc[i2] < df["MACD_Hist"].iloc[i1]
        ):
            bear.append((i1, i2))

    return bull[-3:], bear[-3:]
# ============================================================
# UNIFIED INDICATOR ENGINE — compute_indicators()
# ============================================================

def compute_indicators(df: pd.DataFrame, swing_bars: int = 5) -> pd.DataFrame:
    """
    Master indicator engine for Sniper Terminal v4.
    Combines:
    - Core indicators (EMA, RSI, MACD, ATR, BB, KC, Squeeze, Stoch, Vol_Ratio, OFI, Candles)
    - Advanced indicators (AVWAP, VWAP anchors)
    - Swing high/low detectors
    - Divergence detection
    """

    if df is None or df.empty:
        return df

    # -----------------------------------------
    # 1) CORE INDICATORS
    # -----------------------------------------
    df = compute_core_indicators(df)

    # -----------------------------------------
    # 2) ADVANCED INDICATORS (AVWAP + Anchors)
    # -----------------------------------------
    df = compute_avwaps(df)

    # -----------------------------------------
    # 3) SWING POINTS
    # -----------------------------------------
    df["Swing_Low"] = swing_lows(df["Low"], bars=swing_bars)
    df["Swing_High"] = swing_highs(df["High"], bars=swing_bars)

    # -----------------------------------------
    # 4) DIVERGENCES
    # -----------------------------------------
    bull_divs, bear_divs = compute_divergences(df, bars=swing_bars)

    # Store last 3 divergences for reference
    df["Bull_Div"] = False
    df["Bear_Div"] = False

    for i1, i2 in bull_divs:
        df.loc[df.index[i2], "Bull_Div"] = True

    for i1, i2 in bear_divs:
        df.loc[df.index[i2], "Bear_Div"] = True

    # -----------------------------------------
    # 5) CLEANUP
    # -----------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how="all")

    return df
# ============================================================
# SETUP ENGINE 1 — DIVERGENCE
# ============================================================

def scan_stock_divergence(
    ticker: str,
    interval: str,
    use_trend: bool,
    fresh_only: bool,
    swing_bars: int,
    min_q: int,
    total_capital: float,
    risk_pct: float,
):
    df = load_data(ticker, interval)
    if df.empty or len(df) < 100:
        return None

    df = compute_indicators(df, swing_bars=swing_bars)

    bull_divs, _ = compute_divergences(df, bars=swing_bars)
    if not bull_divs:
        return None

    i1, i2 = bull_divs[-1]

    # Fresh signal filter
    if fresh_only and i2 < len(df) - 4:
        return None

    # Weekly trend filter
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
    e200 = float(df["EMA200"].iloc[i2])
    avwap = float(df["AVWAP"].iloc[i2])

    if ep < e200 or ep < avwap or atr <= 0:
        return None

    q = signal_quality(df, i2)
    if q["total"] < min_q:
        return None

    sl = round(ep - 1.5 * atr, 2)
    tp1 = round(ep + 2 * atr, 2)
    tp2 = round(ep + 4 * atr, 2)
    rr = round((tp1 - ep) / (ep - sl), 2)

    risk_amount = total_capital * risk_pct / 100.0
    sl_dist = max(ep - sl, 0.01)
    qty = max(int(risk_amount / sl_dist), 0)

    vr = float(df["Vol_Ratio"].iloc[i2])
    ofi = float(df["OFI"].iloc[i2])

    return {
        "Symbol": ticker.replace(".NS", ""),
        "Ticker": ticker,
        "Date": str(df.index[i2])[:10],
        "Entry": ep,
        "SL": sl,
        "TP1": tp1,
        "TP2": tp2,
        "R_R": rr,
        "Quality": q["total"],
        "Grade": q["grade"],
        "RSI": round(float(df["RSI"].iloc[i2]), 1),
        "Vol_Ratio": round(vr, 2),
        "OFI": round(ofi, 3),
        "i1": i1,
        "i2": i2,
        "Position_Qty": qty,
    }


# ============================================================
# SETUP ENGINE 2 — BB SQUEEZE
# ============================================================

def detect_bb_squeeze(df: pd.DataFrame, squeeze_len: int = 10):
    if not all(col in df.columns for col in ["BB_U", "BB_L", "KC_U", "KC_L"]):
        return pd.Series(False, index=df.index)

    in_sq = (df["BB_U"] < df["KC_U"]) & (df["BB_L"] > df["KC_L"])
    in_sq = in_sq.rolling(squeeze_len).apply(lambda x: 1 if x.all() else 0, raw=True).astype(bool)
    return in_sq


def detect_squeeze_fire(df: pd.DataFrame, in_squeeze: pd.Series):
    sq_fire = pd.Series(False, index=df.index)
    for i in range(1, len(df)):
        if in_squeeze.iloc[i - 1] and not in_squeeze.iloc[i]:
            sq_fire.iloc[i] = True
    return sq_fire


def scan_stock_bb_squeeze(
    ticker: str,
    interval: str,
    use_trend: bool,
    fresh_only: bool,
    swing_bars: int,
    min_q: int,
    total_capital: float,
    risk_pct: float,
    sq_only: bool = True,
):
    df = load_data(ticker, interval)
    if df.empty or len(df) < 100:
        return None

    df = compute_indicators(df, swing_bars=swing_bars)

    in_sq = detect_bb_squeeze(df, squeeze_len=10)
    sq_fire = detect_squeeze_fire(df, in_sq)

    df["In_Squeeze"] = in_sq
    df["Squeeze_Fire"] = sq_fire

    # Find most recent squeeze fire
    idx_candidates = []
    for i in range(len(df) - 1, -1, -1):
        if sq_fire.iloc[i]:
            idx_candidates.append(i)
            break

    # If no fire, optionally allow active squeeze
    if not idx_candidates and not sq_only:
        for i in range(len(df) - 1, -1, -1):
            if in_sq.iloc[i]:
                idx_candidates.append(i)
                break

    if not idx_candidates:
        return None

    i2 = idx_candidates[0]

    if fresh_only and i2 < len(df) - 5:
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
    e200 = float(df["EMA200"].iloc[i2])
    avwap = float(df["AVWAP"].iloc[i2])

    if ep < e200 or ep < avwap or atr <= 0:
        return None

    q = signal_quality(df, i2)
    if q["total"] < min_q:
        return None

    sl = round(ep - 1.2 * atr, 2)
    tp1 = round(ep + 2.2 * atr, 2)
    tp2 = round(ep + 4.0 * atr, 2)
    rr = round((tp1 - ep) / (ep - sl), 2)

    risk_amount = total_capital * risk_pct / 100.0
    sl_dist = max(ep - sl, 0.01)
    qty = max(int(risk_amount / sl_dist), 0)

    vr = float(df["Vol_Ratio"].iloc[i2])
    ofi = float(df["OFI"].iloc[i2])

    return {
        "Symbol": ticker.replace(".NS", ""),
        "Ticker": ticker,
        "Date": str(df.index[i2])[:10],
        "Entry": ep,
        "SL": sl,
        "TP1": tp1,
        "TP2": tp2,
        "R_R": rr,
        "Grade": q["grade"],
        "RSI": round(float(df["RSI"].iloc[i2]), 1),
        "Vol_Ratio": round(vr, 2),
        "OFI": round(ofi, 3),
        "In_Squeeze": bool(in_sq.iloc[i2]),
        "Squeeze_Fire": bool(sq_fire.iloc[i2]),
        "i2": i2,
        "Position_Qty": qty,
    }
# ============================================================
# WEEKLY TREND & SIGNAL QUALITY ENGINE
# ============================================================

@st.cache_data(show_spinner=False)
def get_weekly_trend(ticker: str):
    """
    Weekly trend filter:
    - Price above EMA200
    - Weekly RSI > 50
    """
    dfw = load_data(ticker, "1wk", years=5)
    if dfw is None or dfw.empty:
        return None

    c = dfw["Close"]
    dfw["EMA200"] = c.ewm(span=200, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    dfw["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    return (dfw["Close"] > dfw["EMA200"]) & (dfw["RSI"] > 50)


def signal_quality(df: pd.DataFrame, i: int):
    """
    Compact quality score used across setups.
    Uses:
    - RSI zone
    - Volume expansion
    - ATR vs its own history
    - Price vs EMA200
    """
    score = 0
    rsi_val = float(df["RSI"].iloc[i])
    vr = float(df["Vol_Ratio"].iloc[i])
    atr_val = float(df["ATR"].iloc[i])

    # RSI: constructive 40–70
    if 40 <= rsi_val <= 70:
        score += 25

    # Volume: >1.2x 20‑bar average
    if vr > 1.2:
        score += 25

    # ATR: above its 50‑bar mean
    atr_ma50 = df["ATR"].rolling(50).mean().iloc[i]
    if not np.isnan(atr_ma50) and atr_val > atr_ma50:
        score += 25

    # Trend: above EMA200
    if float(df["Close"].iloc[i]) > float(df["EMA200"].iloc[i]):
        score += 25

    if score >= 80:
        grade = "A"
    elif score >= 60:
        grade = "B"
    elif score >= 40:
        grade = "C"
    else:
        grade = "D"

    return {"total": score, "grade": grade}


# ============================================================
# SETUP ENGINE 3 — HIGH VOLUME BREAKOUT
# ============================================================

def find_recent_resistance(df: pd.DataFrame, lookback: int = 40):
    """
    Finds a recent swing‑high resistance level in the last `lookback` bars.
    """
    if len(df) < lookback + 5:
        return None, None

    highs = df["High"].iloc[-lookback:]
    idx_local = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)].index
    if len(idx_local) == 0:
        return None, None

    i_res = idx_local[-1]
    level = float(df.loc[i_res, "High"])
    return level, i_res


def scan_stock_breakout(
    ticker: str,
    interval: str,
    use_trend: bool,
    fresh_only: bool,
    swing_bars: int,
    min_q: int,
    total_capital: float,
    risk_pct: float,
):
    df = load_data(ticker, interval)
    if df.empty or len(df) < 120:
        return None

    df = compute_indicators(df, swing_bars=swing_bars)

    res_level, res_idx = find_recent_resistance(df, lookback=50)
    if res_level is None:
        return None

    i2 = len(df) - 2
    if i2 <= res_idx:
        return None

    close_i2 = float(df["Close"].iloc[i2])
    vol_i2 = float(df["Volume"].iloc[i2])
    vol_ma20 = float(df["Volume"].rolling(20).mean().iloc[i2])

    if vol_ma20 <= 0:
        return None

    vol_ratio_val = vol_i2 / vol_ma20

    # Must close above resistance with strong volume
    if close_i2 <= res_level:
        return None
    if vol_ratio_val < 2.0:
        return None

    if fresh_only and i2 < len(df) - 4:
        return None

    if use_trend:
        tw = get_weekly_trend(ticker)
        if tw is None:
            return None
        if not bool(tw.reindex(df.index, method="ffill").iloc[i2]):
            return None

    e200 = float(df["EMA200"].iloc[i2])
    avwap = float(df["AVWAP"].iloc[i2])
    if close_i2 < e200 or close_i2 < avwap:
        return None

    atr = float(df["ATR"].iloc[i2])
    if atr <= 0:
        return None

    ei = i2 + 1
    if ei >= len(df):
        return None
    ep = float(df["Open"].iloc[ei])

    q = signal_quality(df, i2)
    if q["total"] < min_q:
        return None

    sl = round(res_level - 0.8 * atr, 2)
    tp1 = round(ep + 2.0 * atr, 2)
    tp2 = round(ep + 4.0 * atr, 2)
    rr = round((tp1 - ep) / (ep - sl), 2) if ep != sl else 0

    risk_amount = total_capital * risk_pct / 100.0
    sl_dist = max(ep - sl, 0.01)
    qty = max(int(risk_amount / sl_dist), 0)

    vr = float(df["Vol_Ratio"].iloc[i2])
    ofi = float(df["OFI"].iloc[i2])

    return {
        "Symbol": ticker.replace(".NS", ""),
        "Ticker": ticker,
        "Date": str(df.index[i2])[:10],
        "Entry": ep,
        "SL": sl,
        "TP1": tp1,
        "TP2": tp2,
        "R_R": rr,
        "Grade": q["grade"],
        "RSI": round(float(df["RSI"].iloc[i2]), 1),
        "Vol_Ratio": round(vr, 2),
        "OFI": round(ofi, 3),
        "Breakout_Level": round(res_level, 2),
        "i2": i2,
        "Position_Qty": qty,
    }


# ============================================================
# SETUP ENGINE 4 — TREND PULLBACK
# ============================================================

def detect_reversal_candle(df: pd.DataFrame, i: int):
    """
    Simple bullish reversal candle:
    - Green body
    - Lower wick > body and > upper wick
    """
    if i <= 0 or i >= len(df):
        return False

    o = float(df["Open"].iloc[i])
    c = float(df["Close"].iloc[i])
    h = float(df["High"].iloc[i])
    l = float(df["Low"].iloc[i])

    body = abs(c - o)
    if body <= 0:
        return False

    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)

    if c > o and lower_wick > body * 1.2 and lower_wick > upper_wick:
        return True
    return False


def scan_stock_pullback(
    ticker: str,
    interval: str,
    use_trend: bool,
    fresh_only: bool,
    swing_bars: int,
    min_q: int,
    total_capital: float,
    risk_pct: float,
):
    df = load_data(ticker, interval)
    if df.empty or len(df) < 120:
        return None

    df = compute_indicators(df, swing_bars=swing_bars)

    i2 = len(df) - 2
    if i2 < 20:
        return None

    ema21 = float(df["EMA21"].iloc[i2])
    ema50 = float(df["EMA50"].iloc[i2])
    ema200 = float(df["EMA200"].iloc[i2])
    close_i2 = float(df["Close"].iloc[i2])

    # Strong uptrend
    if not (close_i2 > ema21 and close_i2 > ema50 and close_i2 > ema200):
        return None

    # Recent lows near EMA21/50
    recent_lows = df["Low"].iloc[i2 - 5 : i2 + 1]
    near_ema = (
        (abs(recent_lows - ema21) / ema21 < 0.02) |
        (abs(recent_lows - ema50) / ema50 < 0.02)
    )
    if not near_ema.any():
        return None

    # Reversal candle at i2
    if not detect_reversal_candle(df, i2):
        return None

    # Volume confirmation: reversal bar > 1.2x pullback volume
    vol = df["Volume"]
    vol_pullback = vol.iloc[i2 - 5 : i2]
    vol_rev = vol.iloc[i2]

    if vol_pullback.mean() <= 0:
        return None
    if not (vol_rev > vol_pullback.mean() * 1.2):
        return None

    if fresh_only and i2 < len(df) - 4:
        return None

    if use_trend:
        tw = get_weekly_trend(ticker)
        if tw is None:
            return None
        if not bool(tw.reindex(df.index, method="ffill").iloc[i2]):
            return None

    atr = float(df["ATR"].iloc[i2])
    if atr <= 0:
        return None

    ei = i2 + 1
    if ei >= len(df):
        return None
    ep = float(df["Open"].iloc[ei])

    q = signal_quality(df, i2)
    if q["total"] < min_q:
        return None

    sl = round(float(df["Low"].iloc[i2]) - 0.5 * atr, 2)
    tp1 = round(ep + 2.0 * atr, 2)
    tp2 = round(ep + 4.0 * atr, 2)
    rr = round((tp1 - ep) / (ep - sl), 2) if ep != sl else 0

    risk_amount = total_capital * risk_pct / 100.0
    sl_dist = max(ep - sl, 0.01)
    qty = max(int(risk_amount / sl_dist), 0)

    vr = float(df["Vol_Ratio"].iloc[i2])
    ofi = float(df["OFI"].iloc[i2])

    return {
        "Symbol": ticker.replace(".NS", ""),
        "Ticker": ticker,
        "Date": str(df.index[i2])[:10],
        "Entry": ep,
        "SL": sl,
        "TP1": tp1,
        "TP2": tp2,
        "R_R": rr,
        "Grade": q["grade"],
        "RSI": round(float(df["RSI"].iloc[i2]), 1),
        "Vol_Ratio": round(vr, 2),
        "OFI": round(ofi, 3),
        "EMA_Touch": "EMA21/50 Pullback",
        "Reversal_Candle": True,
        "i2": i2,
        "Position_Qty": qty,
    }
# ============================================================
# SETUP ENGINE 5 — LIQUIDITY SWEEP (STOP-HUNT REVERSAL)
# ============================================================

def is_stop_hunt_wick(df: pd.DataFrame, i: int, side: str = "bull"):
    """
    Detects a stop-hunt wick:
    - Long wick against previous candle extreme
    - Body small relative to wick
    - Close back inside previous range
    """
    if i <= 1 or i >= len(df):
        return False

    o = float(df["Open"].iloc[i])
    c = float(df["Close"].iloc[i])
    h = float(df["High"].iloc[i])
    l = float(df["Low"].iloc[i])

    body = abs(c - o)
    if body <= 0:
        return False

    if side == "bull":
        # Bullish sweep: long lower wick
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        prev_low = float(df["Low"].iloc[i - 1])

        if (
            lower_wick > body * 1.5 and
            lower_wick > upper_wick and
            c > prev_low
        ):
            return True

    else:
        # Bearish sweep: long upper wick
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        prev_high = float(df["High"].iloc[i - 1])

        if (
            upper_wick > body * 1.5 and
            upper_wick > lower_wick and
            c < prev_high
        ):
            return True

    return False


def scan_stock_sweep(
    ticker: str,
    interval: str,
    use_trend: bool,
    fresh_only: bool,
    swing_bars: int,
    min_q: int,
    total_capital: float,
    risk_pct: float,
):
    df = load_data(ticker, interval)
    if df.empty or len(df) < 80:
        return None

    df = compute_indicators(df, swing_bars=swing_bars)

    i2 = len(df) - 2
    if i2 < 5:
        return None

    # Must form a bullish stop-hunt wick
    if not is_stop_hunt_wick(df, i2, side="bull"):
        return None

    # Volume confirmation
    vol = df["Volume"]
    vol_ma20 = vol.rolling(20).mean()
    if vol_ma20.iloc[i2] <= 0:
        return None
    if vol.iloc[i2] < vol_ma20.iloc[i2] * 1.5:
        return None

    # Order flow must flip positive
    ofi = float(df["OFI"].iloc[i2])
    if np.isnan(ofi) or ofi <= 0:
        return None

    if fresh_only and i2 < len(df) - 4:
        return None

    # Weekly trend filter
    if use_trend:
        tw = get_weekly_trend(ticker)
        if tw is None:
            return None
        if not bool(tw.reindex(df.index, method="ffill").iloc[i2]):
            return None

    atr = float(df["ATR"].iloc[i2])
    if atr <= 0:
        return None

    ei = i2 + 1
    if ei >= len(df):
        return None

    ep = float(df["Open"].iloc[ei])

    q = signal_quality(df, i2)
    if q["total"] < min_q:
        return None

    sl = round(float(df["Low"].iloc[i2]) - 0.3 * atr, 2)
    tp1 = round(ep + 2.5 * atr, 2)
    tp2 = round(ep + 5.0 * atr, 2)
    rr = round((tp1 - ep) / (ep - sl), 2)

    risk_amount = total_capital * risk_pct / 100.0
    sl_dist = max(ep - sl, 0.01)
    qty = max(int(risk_amount / sl_dist), 0)

    vr = float(df["Vol_Ratio"].iloc[i2])

    return {
        "Symbol": ticker.replace(".NS", ""),
        "Ticker": ticker,
        "Date": str(df.index[i2])[:10],
        "Entry": ep,
        "SL": sl,
        "TP1": tp1,
        "TP2": tp2,
        "R_R": rr,
        "Grade": q["grade"],
        "RSI": round(float(df["RSI"].iloc[i2]), 1),
        "Vol_Ratio": round(vr, 2),
        "OFI": round(ofi, 3),
        "Sweep_Level": round(float(df["Low"].iloc[i2]), 2),
        "i2": i2,
        "Position_Qty": qty,
    }


# ============================================================
# SETUP ENGINE 6 — VCP (VOLATILITY CONTRACTION PATTERN)
# ============================================================

def compute_contraction_series(df: pd.DataFrame, lookback: int = 60):
    """
    Detects volatility contractions using pivot highs/lows.
    Returns last 3 contraction ranges.
    """
    if len(df) < lookback + 10:
        return []

    sub = df.iloc[-lookback:]
    highs = sub["High"]
    lows = sub["Low"]

    swing_high_idx = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)].index
    swing_low_idx = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)].index

    pivots = sorted(list(swing_high_idx) + list(swing_low_idx))
    if len(pivots) < 4:
        return []

    contractions = []
    for i in range(2, len(pivots)):
        i_start = pivots[i - 2]
        i_end = pivots[i]
        h = df.loc[i_start:i_end, "High"].max()
        l = df.loc[i_start:i_end, "Low"].min()
        if l <= 0:
            continue
        pct_range = (h - l) / l * 100.0
        contractions.append((i_start, i_end, pct_range))

    return contractions[-3:]


def scan_stock_vcp(
    ticker: str,
    interval: str,
    use_trend: bool,
    fresh_only: bool,
    swing_bars: int,
    min_q: int,
    total_capital: float,
    risk_pct: float,
):
    df = load_data(ticker, interval)
    if df.empty or len(df) < 150:
        return None

    df = compute_indicators(df, swing_bars=swing_bars)

    contractions = compute_contraction_series(df, lookback=80)
    if len(contractions) < 2:
        return None

    pct_ranges = [c[2] for c in contractions]

    # Must show contraction: last < previous
    if not (pct_ranges[-1] < pct_ranges[-2]):
        return None

    _, i_pivot, _ = contractions[-1]
    if i_pivot >= len(df) - 3:
        return None

    pivot_high = df["High"].loc[:i_pivot].max()

    # Volume dry-up near pivot
    vol = df["Volume"]
    vol_pivot = vol.loc[i_pivot - 10 : i_pivot]
    vol_prev = vol.loc[i_pivot - 30 : i_pivot - 11] if i_pivot - 30 >= 0 else None

    if vol_prev is None or len(vol_prev) < 5:
        return None

    if vol_pivot.mean() >= vol_prev.mean():
        return None

    i2 = len(df) - 2
    if i2 <= i_pivot:
        return None

    close_i2 = float(df["Close"].iloc[i2])
    if close_i2 <= pivot_high:
        return None

    vol_ma20 = vol.rolling(20).mean()
    if vol_ma20.iloc[i2] <= 0:
        return None
    if vol.iloc[i2] < vol_ma20.iloc[i2] * 1.5:
        return None

    if fresh_only and i2 < len(df) - 4:
        return None

    if use_trend:
        tw = get_weekly_trend(ticker)
        if tw is None:
            return None
        if not bool(tw.reindex(df.index, method="ffill").iloc[i2]):
            return None

    e200 = float(df["EMA200"].iloc[i2])
    avwap = float(df["AVWAP"].iloc[i2])
    if close_i2 < e200 or close_i2 < avwap:
        return None

    atr = float(df["ATR"].iloc[i2])
    if atr <= 0:
        return None

    ei = i2 + 1
    if ei >= len(df):
        return None

    ep = float(df["Open"].iloc[ei])

    q = signal_quality(df, i2)
    if q["total"] < min_q:
        return None

    sl = round(pivot_high - 0.8 * atr, 2)
    tp1 = round(ep + 2.5 * atr, 2)
    tp2 = round(ep + 5.0 * atr, 2)
    rr = round((tp1 - ep) / (ep - sl), 2)

    risk_amount = total_capital * risk_pct / 100.0
    sl_dist = max(ep - sl, 0.01)
    qty = max(int(risk_amount / sl_dist), 0)

    vr = float(df["Vol_Ratio"].iloc[i2])
    ofi = float(df["OFI"].iloc[i2])
    contraction_pct = pct_ranges[-1]

    return {
        "Symbol": ticker.replace(".NS", ""),
        "Ticker": ticker,
        "Date": str(df.index[i2])[:10],
        "Entry": ep,
        "SL": sl,
        "TP1": tp1,
        "TP2": tp2,
        "R_R": rr,
        "Grade": q["grade"],
        "RSI": round(float(df["RSI"].iloc[i2]), 1),
        "Vol_Ratio": round(vr, 2),
        "OFI": round(ofi, 3),
        "Contraction_Pct": round(contraction_pct, 2),
        "Vol_Dryup": True,
        "i2": i2,
        "Position_Qty": qty,
    }
# ============================================================
# SMART COLUMNS FOR EACH SETUP
# ============================================================

def get_screener_columns_for_setup(setup_type: str):
    if setup_type == "Divergence":
        return [
            "Symbol", "Ticker", "Date",
            "Entry", "SL", "TP1", "TP2",
            "R_R", "Quality", "Grade",
            "RSI", "Vol_Ratio", "OFI",
        ]

    elif setup_type == "BB Squeeze":
        return [
            "Symbol", "Ticker", "Date",
            "Entry", "SL", "TP1", "TP2",
            "R_R", "Grade",
            "In_Squeeze", "Squeeze_Fire",
            "Vol_Ratio", "RSI",
        ]

    elif setup_type == "High Volume Breakout":
        return [
            "Symbol", "Ticker", "Date",
            "Entry", "SL", "TP1", "TP2",
            "Breakout_Level",
            "R_R", "Grade",
            "Vol_Ratio", "RSI", "OFI",
        ]

    elif setup_type == "Trend Pullback":
        return [
            "Symbol", "Ticker", "Date",
            "Entry", "SL", "TP1", "TP2",
            "EMA_Touch", "Reversal_Candle",
            "R_R", "Grade",
            "Vol_Ratio", "RSI", "OFI",
        ]

    elif setup_type == "Liquidity Sweep":
        return [
            "Symbol", "Ticker", "Date",
            "Entry", "SL", "TP1", "TP2",
            "Sweep_Level",
            "R_R", "Grade",
            "Vol_Ratio", "RSI", "OFI",
        ]

    elif setup_type == "VCP Pattern":
        return [
            "Symbol", "Ticker", "Date",
            "Entry", "SL", "TP1", "TP2",
            "Contraction_Pct", "Vol_Dryup",
            "R_R", "Grade",
            "Vol_Ratio", "RSI", "OFI",
        ]

    return ["Symbol", "Ticker", "Date", "Entry", "SL", "R_R", "Grade"]


# ============================================================
# SETUP-SPECIFIC ANALYSIS TEXT
# ============================================================

def explain_divergence(row, df, i2):
    return f"""
### Divergence Setup Analysis

- Price made a **lower low**, but RSI & MACD showed **higher lows** → bullish divergence.
- Divergence confirmed at index **{i2}** on **{row['Date']}**.
- RSI: **{row.get('RSI')}**, Volume Ratio: **{row.get('Vol_Ratio')}**, OFI: **{row.get('OFI')}**.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Quality Score: **{row.get('Quality')}**, Grade: **{row.get('Grade')}**.

This is a **high‑probability reversal setup** with strong momentum shift.
"""


def explain_bb_squeeze(row, df, i2):
    return f"""
### BB Squeeze Setup Analysis

- Bollinger Bands contracted **inside** Keltner Channels → volatility compression.
- In Squeeze: **{row.get('In_Squeeze')}**, Squeeze Fire: **{row.get('Squeeze_Fire')}**.
- Volume Ratio: **{row.get('Vol_Ratio')}**, RSI: **{row.get('RSI')}**.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Grade: **{row.get('Grade')}**.

This is a **continuation breakout** after volatility contraction.
"""


def explain_breakout(row, df, i2):
    return f"""
### High Volume Breakout Analysis

- Price broke above resistance: **{row.get('Breakout_Level')}**.
- Volume Ratio: **{row.get('Vol_Ratio')}** → confirms strong participation.
- RSI: **{row.get('RSI')}**, OFI: **{row.get('OFI')}**.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Grade: **{row.get('Grade')}**.

This is a **momentum breakout** with strong volume confirmation.
"""


def explain_pullback(row, df, i2):
    return f"""
### Trend Pullback Analysis

- Price pulled back to **EMA21/50** and formed a bullish reversal candle.
- Volume on reversal > 1.2× pullback volume.
- RSI: **{row.get('RSI')}**, Volume Ratio: **{row.get('Vol_Ratio')}**.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Grade: **{row.get('Grade')}**.

This is a **trend continuation setup** with clean structure.
"""


def explain_sweep(row, df, i2):
    return f"""
### Liquidity Sweep Analysis

- Price swept liquidity below **{row.get('Sweep_Level')}** and reversed strongly.
- OFI positive → buyers stepped in aggressively.
- Volume Ratio: **{row.get('Vol_Ratio')}**, RSI: **{row.get('RSI')}**.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Grade: **{row.get('Grade')}**.

This is a **stop-hunt reversal** with strong order-flow confirmation.
"""


def explain_vcp(row, df, i2):
    return f"""
### VCP Pattern Analysis

- Volatility contraction detected: **{row.get('Contraction_Pct')}%**.
- Volume dry-up confirmed near pivot.
- Breakout above pivot high with strong volume.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Grade: **{row.get('Grade')}**.

This is a **classic VCP breakout** with contraction + volume expansion.
"""


# ============================================================
# SETUP DISPATCHER
# ============================================================

def run_setup_scan(
    setup_type: str,
    ticker: str,
    interval: str,
    use_trend: bool,
    fresh_only: bool,
    swing_bars: int,
    min_q: int,
    total_capital: float,
    risk_pct: float,
):
    if setup_type == "Divergence":
        return scan_stock_divergence(
            ticker, interval, use_trend, fresh_only,
            swing_bars, min_q, total_capital, risk_pct
        )

    elif setup_type == "BB Squeeze":
        return scan_stock_bb_squeeze(
            ticker, interval, use_trend, fresh_only,
            swing_bars, min_q, total_capital, risk_pct
        )

    elif setup_type == "High Volume Breakout":
        return scan_stock_breakout(
            ticker, interval, use_trend, fresh_only,
            swing_bars, min_q, total_capital, risk_pct
        )

    elif setup_type == "Trend Pullback":
        return scan_stock_pullback(
            ticker, interval, use_trend, fresh_only,
            swing_bars, min_q, total_capital, risk_pct
        )

    elif setup_type == "Liquidity Sweep":
        return scan_stock_sweep(
            ticker, interval, use_trend, fresh_only,
            swing_bars, min_q, total_capital, risk_pct
        )

    elif setup_type == "VCP Pattern":
        return scan_stock_vcp(
            ticker, interval, use_trend, fresh_only,
            swing_bars, min_q, total_capital, risk_pct
        )

    return None


# ============================================================
# SCREENER ENGINE — RUN ACROSS UNIVERSE
# ============================================================

def run_screener(
    setup_type: str,
    universe: list,
    interval: str,
    use_trend: bool,
    fresh_only: bool,
    swing_bars: int,
    min_q: int,
    total_capital: float,
    risk_pct: float,
):
    results = []

    for ticker in universe:
        try:
            row = run_setup_scan(
                setup_type, ticker, interval,
                use_trend, fresh_only,
                swing_bars, min_q,
                total_capital, risk_pct
            )
            if row:
                results.append(row)
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Sort by quality → RR → volume ratio
    sort_cols = [col for col in ["Quality", "R_R", "Vol_Ratio"] if col in df.columns]
    df = df.sort_values(sort_cols, ascending=False)

    return df
# ============================================================
# BACKTEST ENGINE — Sniper Terminal v4
# ============================================================

def simulate_trade(entry, sl, tp1, tp2, df_slice):
    """
    Simulates a single trade:
    - Entry at next candle open
    - SL, TP1, TP2 hit detection
    - Returns outcome, exit price, R-multiple
    """
    for i in range(len(df_slice)):
        low = df_slice["Low"].iloc[i]
        high = df_slice["High"].iloc[i]

        # SL hit first
        if low <= sl:
            return "SL", sl, (sl - entry)

        # TP1 hit
        if high >= tp1:
            # Check TP2
            if high >= tp2:
                return "TP2", tp2, (tp2 - entry)
            return "TP1", tp1, (tp1 - entry)

    # No target hit → exit at last close
    last_close = df_slice["Close"].iloc[-1]
    return "EXIT", last_close, (last_close - entry)


def backtest_setup(
    df: pd.DataFrame,
    signals: list,
    risk_pct: float,
    total_capital: float,
):
    """
    Backtests a list of signals:
    - Each signal contains Entry, SL, TP1, TP2, i2 index
    - Uses simulate_trade() for outcome
    """
    trades = []
    capital = total_capital

    for sig in signals:
        i2 = sig["i2"]
        if i2 + 1 >= len(df):
            continue

        entry = sig["Entry"]
        sl = sig["SL"]
        tp1 = sig["TP1"]
        tp2 = sig["TP2"]

        df_slice = df.iloc[i2 + 1 : i2 + 40]  # 40 bars lookahead
        if df_slice.empty:
            continue

        outcome, exit_price, pnl = simulate_trade(entry, sl, tp1, tp2, df_slice)

        risk_amount = capital * risk_pct / 100.0
        sl_dist = max(entry - sl, 0.01)
        qty = max(int(risk_amount / sl_dist), 1)

        profit = qty * pnl
        capital += profit

        trades.append({
            "Date": sig["Date"],
            "Entry": entry,
            "Exit": exit_price,
            "Outcome": outcome,
            "PnL": round(profit, 2),
            "Capital": round(capital, 2),
            "R_Multiple": round(pnl / sl_dist, 2),
        })

    return pd.DataFrame(trades)
# ============================================================
# AUTOTRADE ENGINE — PAPER MODE
# ============================================================

def autotrade_signal_executor(
    df: pd.DataFrame,
    signal: dict,
    risk_pct: float,
    total_capital: float,
):
    """
    Executes a single signal in paper mode.
    """
    i2 = signal["i2"]
    if i2 + 1 >= len(df):
        return None

    entry = signal["Entry"]
    sl = signal["SL"]
    tp1 = signal["TP1"]
    tp2 = signal["TP2"]

    df_slice = df.iloc[i2 + 1 : i2 + 40]
    if df_slice.empty:
        return None

    outcome, exit_price, pnl = simulate_trade(entry, sl, tp1, tp2, df_slice)

    risk_amount = total_capital * risk_pct / 100.0
    sl_dist = max(entry - sl, 0.01)
    qty = max(int(risk_amount / sl_dist), 1)

    profit = qty * pnl

    return {
        "Symbol": signal["Symbol"],
        "Entry": entry,
        "Exit": exit_price,
        "Outcome": outcome,
        "PnL": round(profit, 2),
        "Qty": qty,
        "R_Multiple": round(pnl / sl_dist, 2),
        "Date": signal["Date"],
    }
# ============================================================
# VOLUME PROFILE ENGINE
# ============================================================

def compute_volume_profile(df: pd.DataFrame, bins: int = 24):
    """
    Computes a simple volume profile:
    - Splits price range into bins
    - Sums volume per bin
    """
    if df.empty:
        return None, None, None

    low = df["Low"].min()
    high = df["High"].max()
    edges = np.linspace(low, high, bins + 1)

    vol = np.zeros(bins)
    for i in range(bins):
        mask = (df["Close"] >= edges[i]) & (df["Close"] < edges[i + 1])
        vol[i] = df.loc[mask, "Volume"].sum()

    poc_idx = int(np.argmax(vol))
    poc = (edges[poc_idx] + edges[poc_idx + 1]) / 2

    return edges, vol, poc


# ============================================================
# PLOTLY CHART BUILDER — ROYAL GOLD THEME
# ============================================================

def plot_chart(df: pd.DataFrame, ticker: str, signals=None):
    """
    Builds the main chart:
    - Candles
    - EMA21/50/200
    - AVWAP anchors
    - Divergence markers
    - Volume profile (optional)
    """
    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.03,
    )

    # Candles
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        ),
        row=1, col=1
    )

    # EMAs
    for p, col in [(21, "EMA21"), (50, "EMA50"), (200, "EMA200")]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=col,
                ),
                row=1, col=1
            )

    # AVWAP anchors
    for col in ["vwap_top", "vwap_bot", "vwap_rtop", "vwap_rbot"]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=col,
                    line=dict(width=1, dash="dot"),
                ),
                row=1, col=1
            )

    # Divergence markers
    if "Bull_Div" in df.columns:
        bull_idx = df.index[df["Bull_Div"]]
        fig.add_trace(
            go.Scatter(
                x=bull_idx,
                y=df.loc[bull_idx, "Low"],
                mode="markers",
                marker=dict(color="lime", size=10),
                name="Bull Div",
            ),
            row=1, col=1
        )

    if "Bear_Div" in df.columns:
        bear_idx = df.index[df["Bear_Div"]]
        fig.add_trace(
            go.Scatter(
                x=bear_idx,
                y=df.loc[bear_idx, "High"],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Bear Div",
            ),
            row=1, col=1
        )

    # Volume bars
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
        ),
        row=2, col=1
    )

    fig.update_layout(
        template=st.session_state["plot_theme"],
        height=700,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
    )

    return fig
# ============================================================
# ROYAL GOLD — SIDEBAR NAVIGATION
# ============================================================

def sidebar_navigation():
    st.sidebar.markdown(
        "<div class='royal-title'>Sniper Terminal v4</div>",
        unsafe_allow_html=True
    )

    st.sidebar.markdown(
        "<div class='royal-subtitle'>Royal Gold Edition</div><br>",
        unsafe_allow_html=True
    )

    nav = st.sidebar.radio(
        "Navigation",
        ["Screener", "Chart", "Backtest", "Autotrade", "Volume Profile"],
        index=["Screener", "Chart", "Backtest", "Autotrade", "Volume Profile"].index(
            st.session_state.get("nav_page", "Screener")
        )
    )

    st.session_state["nav_page"] = nav
    return nav


# ============================================================
# SETUP SELECTOR (SEGMENTED CONTROL)
# ============================================================

def setup_selector():
    setups = [
        "Divergence",
        "BB Squeeze",
        "High Volume Breakout",
        "Trend Pullback",
        "Liquidity Sweep",
        "VCP Pattern",
    ]

    selected = st.segmented_control(
        "Select Setup",
        setups,
        default=st.session_state.get("selected_setup", "Divergence")
    )

    st.session_state["selected_setup"] = selected
    return selected


# ============================================================
# SCREENER PAGE
# ============================================================

def page_screener():
    st.markdown("<div class='royal-title'>Screener</div>", unsafe_allow_html=True)

    setup_type = setup_selector()

    col1, col2, col3 = st.columns(3)
    with col1:
        interval = st.selectbox("Interval", ["1d", "1h", "15m"])
    with col2:
        use_trend = st.checkbox("Weekly Trend Filter", True)
    with col3:
        fresh_only = st.checkbox("Fresh Signals Only", True)

    col4, col5, col6 = st.columns(3)
    with col4:
        swing_bars = st.number_input("Swing Bars", 3, 10, 5)
    with col5:
        min_q = st.number_input("Min Quality", 0, 100, 50)
    with col6:
        total_capital = st.number_input("Total Capital", 10000, 10000000, 200000)

    risk_pct = st.slider("Risk % per Trade", 0.1, 5.0, 1.0)

    universe = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
        "SBIN.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "HINDUNILVR.NS",
        "MARUTI.NS", "TITAN.NS", "BAJFINANCE.NS", "ITC.NS", "ULTRACEMCO.NS"
    ]

    if st.button("Run Screener"):
        with st.spinner("Scanning market..."):
            df = run_screener(
                setup_type,
                universe,
                interval,
                use_trend,
                fresh_only,
                swing_bars,
                min_q,
                total_capital,
                risk_pct,
            )

        if df.empty:
            st.warning("No signals found.")
            return

        cols = get_screener_columns_for_setup(setup_type)
        st.dataframe(df[cols], use_container_width=True)

        # Detailed analysis for selected row
        st.markdown("---")
        st.subheader("Setup Analysis")

        selected_row = st.selectbox("Select a signal", df.index)
        row = df.loc[selected_row]

        ticker = row["Ticker"]
        df_full = load_data(ticker, interval)
        df_full = compute_indicators(df_full)

        i2 = int(row["i2"])

        if setup_type == "Divergence":
            st.markdown(explain_divergence(row, df_full, i2))
        elif setup_type == "BB Squeeze":
            st.markdown(explain_bb_squeeze(row, df_full, i2))
        elif setup_type == "High Volume Breakout":
            st.markdown(explain_breakout(row, df_full, i2))
        elif setup_type == "Trend Pullback":
            st.markdown(explain_pullback(row, df_full, i2))
        elif setup_type == "Liquidity Sweep":
            st.markdown(explain_sweep(row, df_full, i2))
        elif setup_type == "VCP Pattern":
            st.markdown(explain_vcp(row, df_full, i2))


# ============================================================
# CHART PAGE
# ============================================================

def page_chart():
    st.markdown("<div class='royal-title'>Chart</div>", unsafe_allow_html=True)

    ticker = st.text_input("Ticker", "RELIANCE.NS")
    interval = st.selectbox("Interval", ["1d", "1h", "15m"])

    if st.button("Load Chart"):
        df = load_data(ticker, interval)
        df = compute_indicators(df)

        fig = plot_chart(df, ticker)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# BACKTEST PAGE
# ============================================================

def page_backtest():
    st.markdown("<div class='royal-title'>Backtest</div>", unsafe_allow_html=True)

    ticker = st.text_input("Ticker", "RELIANCE.NS")
    interval = st.selectbox("Interval", ["1d", "1h"])
    setup_type = setup_selector()

    total_capital = st.number_input("Starting Capital", 10000, 10000000, 200000)
    risk_pct = st.slider("Risk % per Trade", 0.1, 5.0, 1.0)

    if st.button("Run Backtest"):
        df = load_data(ticker, interval)
        df = compute_indicators(df)

        # Generate signals for backtest
        signals = []
        for i in range(20, len(df) - 5):
            row = run_setup_scan(
                setup_type,
                ticker,
                interval,
                use_trend=False,
                fresh_only=False,
                swing_bars=5,
                min_q=40,
                total_capital=total_capital,
                risk_pct=risk_pct,
            )
            if row:
                signals.append(row)

        if not signals:
            st.warning("No signals found for backtest.")
            return

        bt = backtest_setup(df, signals, risk_pct, total_capital)
        st.dataframe(bt, use_container_width=True)

        st.subheader("Equity Curve")
        st.line_chart(bt["Capital"])


# ============================================================
# AUTOTRADE PAGE (PAPER MODE)
# ============================================================

def page_autotrade():
    st.markdown("<div class='royal-title'>Autotrade (Paper)</div>", unsafe_allow_html=True)

    ticker = st.text_input("Ticker", "RELIANCE.NS")
    interval = st.selectbox("Interval", ["1d", "1h", "15m"])
    setup_type = setup_selector()

    total_capital = st.number_input("Capital", 10000, 10000000, 200000)
    risk_pct = st.slider("Risk % per Trade", 0.1, 5.0, 1.0)

    if st.button("Execute Last Signal"):
        df = load_data(ticker, interval)
        df = compute_indicators(df)

        sig = run_setup_scan(
            setup_type,
            ticker,
            interval,
            use_trend=True,
            fresh_only=True,
            swing_bars=5,
            min_q=50,
            total_capital=total_capital,
            risk_pct=risk_pct,
        )

        if not sig:
            st.warning("No valid signal found.")
            return

        trade = autotrade_signal_executor(df, sig, risk_pct, total_capital)
        st.json(trade)


# ============================================================
# VOLUME PROFILE PAGE
# ============================================================

def page_volume_profile():
    st.markdown("<div class='royal-title'>Volume Profile</div>", unsafe_allow_html=True)

    ticker = st.text_input("Ticker", "RELIANCE.NS")
    interval = st.selectbox("Interval", ["1d", "1h"])

    if st.button("Compute Volume Profile"):
        df = load_data(ticker, interval)
        edges, vol, poc = compute_volume_profile(df)

        st.write("POC:", poc)

        st.bar_chart(vol)


# ============================================================
# MAIN APP
# ============================================================

def main():
    apply_theme()
    nav = sidebar_navigation()

    if nav == "Screener":
        page_screener()
    elif nav == "Chart":
        page_chart()
    elif nav == "Backtest":
        page_backtest()
    elif nav == "Autotrade":
        page_autotrade()
    elif nav == "Volume Profile":
        page_volume_profile()


if __name__ == "__main__":
    main()
