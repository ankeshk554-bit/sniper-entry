import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ============================================================
# BASIC CONSTANTS / UNIVERSES
# ============================================================

NIFTY50 = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
]

NIFTY200 = NIFTY50  # extend as you like

SETUP_TYPES = [
    "Divergence",
    "BB Squeeze",
    "High Volume Breakout",
    "Trend Pullback",
    "Liquidity Sweep",
    "VCP Pattern",
]

# ============================================================
# THEME / CSS
# ============================================================

def apply_theme():
    st.markdown(
        """
        <style>
        :root {
            --bg: #050509;
            --card: #101018;
            --fg: #f5f5f7;
            --muted: #9b9bb5;
            --gold: #f7e7ce;
            --gold-soft: #d9c29c;
            --border: #2b2b3d;
            --accent: #ffb347;
        }
        body {
            background-color: var(--bg);
            color: var(--fg);
        }
        .main {
            background-color: var(--bg);
        }
        .royal-card {
            background: radial-gradient(circle at top, rgba(247,231,206,0.08), #050509);
            border-radius: 16px;
            border: 1px solid var(--border);
            padding: 16px 18px;
            margin-bottom: 12px;
        }
        .setup-segment {
            display: inline-flex;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: radial-gradient(circle at top, rgba(247,231,206,0.12), rgba(0,0,0,0.6));
            padding: 2px;
            margin-bottom: 10px;
        }
        .setup-segment button {
            border: none;
            background: transparent;
            color: var(--fg);
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            cursor: pointer;
        }
        .setup-segment button.active {
            background: linear-gradient(135deg, var(--gold), var(--gold-soft));
            color: #111;
            box-shadow: 0 0 12px rgba(247,231,206,0.45);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# DATA & INDICATORS
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(ticker: str, interval: str):
    period = "1y"
    if interval in ["1h", "15m"]:
        period = "60d"
    elif interval == "1wk":
        period = "5y"
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def rsi(series, length=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd_hist(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd = fast_ema - slow_ema
    signal_line = ema(macd, signal)
    return macd - signal_line


def atr(df, length=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def avwap(df):
    pv = (df["Close"] * df["Volume"]).cumsum()
    vol_cum = df["Volume"].cumsum().replace(0, np.nan)
    return pv / vol_cum


def vol_ratio(df, length=20):
    v = df["Volume"]
    ma = v.rolling(length).mean()
    return v / ma.replace(0, np.nan)


def ofi(df):
    o = df["Open"]
    c = df["Close"]
    v = df["Volume"]
    prev_c = c.shift()
    ofi_val = np.where(
        c > prev_c,
        v,
        np.where(c < prev_c, -v, 0),
    )
    return pd.Series(ofi_val, index=df.index)


def bb_kc(df, length=20, mult_bb=2.0, mult_kc=1.5):
    mid = df["Close"].rolling(length).mean()
    std = df["Close"].rolling(length).std()
    bb_upper = mid + mult_bb * std
    bb_lower = mid - mult_bb * std

    tr = atr(df, length)
    kc_upper = mid + mult_kc * tr
    kc_lower = mid - mult_kc * tr

    return bb_upper, bb_lower, kc_upper, kc_lower


def compute_indicators(df: pd.DataFrame):
    df = df.copy()
    df["EMA21"] = ema(df["Close"], 21)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI"] = rsi(df["Close"], 14)
    df["MACD_Hist"] = macd_hist(df["Close"])
    df["ATR"] = atr(df, 14)
    df["AVWAP"] = avwap(df)
    df["Vol_Ratio"] = vol_ratio(df, 20)
    df["OFI"] = ofi(df)
    bb_u, bb_l, kc_u, kc_l = bb_kc(df)
    df["BB_Upper"] = bb_u
    df["BB_Lower"] = bb_l
    df["KC_Upper"] = kc_u
    df["KC_Lower"] = kc_l
    df.dropna(inplace=True)
    return df

# ============================================================
# SWING / TREND / QUALITY
# ============================================================

def swing_lows(series, bars=5):
    return (series.shift(1).rolling(bars).min() > series) & (series.shift(-1).rolling(bars).min() > series)


def swing_highs(series, bars=5):
    return (series.shift(1).rolling(bars).max() < series) & (series.shift(-1).rolling(bars).max() < series)


@st.cache_data(show_spinner=False)
def get_weekly_trend(ticker: str):
    dfw = yf.download(ticker, period="5y", interval="1wk", auto_adjust=False, progress=False)
    if dfw.empty:
        return None
    dfw["EMA200"] = ema(dfw["Close"], 200)
    trend = dfw["Close"] > dfw["EMA200"]
    return trend


def signal_quality(df: pd.DataFrame, i: int):
    score = 0
    rsi_val = df["RSI"].iloc[i]
    vr = df["Vol_Ratio"].iloc[i]
    atr_val = df["ATR"].iloc[i]

    if 40 <= rsi_val <= 70:
        score += 25
    if vr > 1.2:
        score += 25
    if atr_val > df["ATR"].rolling(50).mean().iloc[i]:
        score += 25
    if df["Close"].iloc[i] > df["EMA200"].iloc[i]:
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
# SMART COLUMNS
# ============================================================

def get_screener_columns_for_setup(setup_type: str):
    if setup_type == "Divergence":
        return [
            "Symbol", "Ticker", "Date",
            "Entry", "SL", "TP1", "TP2",
            "R_R", "Quality", "Grade",
            "RSI", "Vol_Ratio",
        ]
    elif setup_type == "BB Squeeze":
        return [
            "Symbol", "Ticker", "Date",
            "Entry", "SL",
            "Squeeze_Fire", "In_Squeeze",
            "Vol_Ratio", "R_R", "Grade",
        ]
    elif setup_type == "High Volume Breakout":
        return [
            "Symbol", "Ticker", "Date",
            "Entry", "SL",
            "Breakout_Level", "Vol_Ratio",
            "R_R", "Grade",
        ]
    elif setup_type == "Trend Pullback":
        return [
            "Symbol", "Ticker", "Date",
            "Entry", "SL",
            "EMA_Touch", "Reversal_Candle",
            "R_R", "Grade",
        ]
    elif setup_type == "Liquidity Sweep":
        return [
            "Symbol", "Ticker", "Date",
            "Entry", "SL",
            "Sweep_Level", "OFI",
            "R_R", "Grade",
        ]
    elif setup_type == "VCP Pattern":
        return [
            "Symbol", "Ticker", "Date",
            "Entry", "SL",
            "Contraction_Pct", "Vol_Dryup",
            "R_R", "Grade",
        ]
    else:
        return [
            "Symbol", "Ticker", "Date",
            "Entry", "SL", "R_R", "Grade",
        ]

# ============================================================
# SETUP MODULE 1 — DIVERGENCE
# ============================================================

def detect_divergence(df: pd.DataFrame, bars=5):
    lm = swing_lows(df["Low"], bars)
    hm = swing_highs(df["High"], bars)

    li = np.where(lm)[0]
    hi = np.where(hm)[0]

    bull, bear = [], []

    for j in range(1, len(li)):
        i1, i2 = li[j - 1], li[j]
        if (
            df["Low"].iloc[i2] < df["Low"].iloc[i1] and
            df["RSI"].iloc[i2] > df["RSI"].iloc[i1] and
            df["MACD_Hist"].iloc[i2] > df["MACD_Hist"].iloc[i1]
        ):
            bull.append((i1, i2))

    for j in range(1, len(hi)):
        i1, i2 = hi[j - 1], hi[j]
        if (
            df["High"].iloc[i2] > df["High"].iloc[i1] and
            df["RSI"].iloc[i2] < df["RSI"].iloc[i1] and
            df["MACD_Hist"].iloc[i2] < df["MACD_Hist"].iloc[i1]
        ):
            bear.append((i1, i2))

    return bull[-3:], bear[-3:]


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

    df = compute_indicators(df)
    bull_divs, _ = detect_divergence(df, bars=swing_bars)

    if not bull_divs:
        return None

    i1, i2 = bull_divs[-1]

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
    atr_val = float(df["ATR"].iloc[i2])
    e200 = float(df["EMA200"].iloc[i2])
    avwap_val = float(df["AVWAP"].iloc[i2])

    if ep < e200 or ep < avwap_val or atr_val <= 0:
        return None

    q = signal_quality(df, i2)
    if q["total"] < min_q:
        return None

    sl = round(ep - 1.5 * atr_val, 2)
    tp1 = round(ep + 2 * atr_val, 2)
    tp2 = round(ep + 4 * atr_val, 2)
    rr = round((tp1 - ep) / (ep - sl), 2) if ep != sl else 0

    risk_amount = total_capital * risk_pct / 100.0
    sl_dist = max(ep - sl, 0.01)
    pos_qty = int(risk_amount / sl_dist)
    pos_qty = max(pos_qty, 0)

    vr_raw = df["Vol_Ratio"].iloc[i2]
    vr = float(vr_raw) if not np.isnan(vr_raw) else 1.0

    ofi_raw = df["OFI"].iloc[i2]
    ofi_val = float(ofi_raw) if not np.isnan(ofi_raw) else 0.0

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
        "OFI": round(ofi_val, 3),
        "i1": i1,
        "i2": i2,
        "Position_Qty": pos_qty,
    }

# ============================================================
# SETUP MODULE 2 — BB SQUEEZE
# ============================================================

def detect_bb_squeeze(df: pd.DataFrame, squeeze_len: int = 10):
    if not all(col in df.columns for col in ["BB_Upper", "BB_Lower", "KC_Upper", "KC_Lower"]):
        return pd.Series(False, index=df.index)

    in_sq = (df["BB_Upper"] < df["KC_Upper"]) & (df["BB_Lower"] > df["KC_Lower"])
    in_sq = in_sq.rolling(squeeze_len).apply(lambda x: 1 if x.all() else 0, raw=True).astype(bool)
    return in_sq


def detect_squeeze_fire(df: pd.DataFrame, in_squeeze: pd.Series):
    squeeze_fire = pd.Series(False, index=df.index)
    for i in range(1, len(df)):
        if in_squeeze.iloc[i - 1] and not in_squeeze.iloc[i]:
            squeeze_fire.iloc[i] = True
    return squeeze_fire


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

    df = compute_indicators(df)

    in_squeeze = detect_bb_squeeze(df, squeeze_len=10)
    squeeze_fire = detect_squeeze_fire(df, in_squeeze)

    df["In_Squeeze"] = in_squeeze
    df["Squeeze_Fire"] = squeeze_fire

    idx_candidates = []

    for i in range(len(df) - 1, -1, -1):
        if squeeze_fire.iloc[i]:
            idx_candidates.append(i)
            break

    if not idx_candidates and not sq_only:
        for i in range(len(df) - 1, -1, -1):
            if in_squeeze.iloc[i]:
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
    atr_val = float(df["ATR"].iloc[i2])
    e200 = float(df["EMA200"].iloc[i2])
    avwap_val = float(df["AVWAP"].iloc[i2])

    if ep < e200 or ep < avwap_val or atr_val <= 0:
        return None

    q = signal_quality(df, i2)
    if q["total"] < min_q:
        return None

    sl = round(ep - 1.2 * atr_val, 2)
    tp1 = round(ep + 2.2 * atr_val, 2)
    tp2 = round(ep + 4.0 * atr_val, 2)
    rr = round((tp1 - ep) / (ep - sl), 2) if ep != sl else 0

    risk_amount = total_capital * risk_pct / 100.0
    sl_dist = max(ep - sl, 0.01)
    pos_qty = int(risk_amount / sl_dist)
    pos_qty = max(pos_qty, 0)

    vr_raw = df["Vol_Ratio"].iloc[i2]
    vr = float(vr_raw) if not np.isnan(vr_raw) else 1.0

    ofi_raw = df["OFI"].iloc[i2]
    ofi_val = float(ofi_raw) if not np.isnan(ofi_raw) else 0.0

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
        "OFI": round(ofi_val, 3),
        "In_Squeeze": bool(in_squeeze.iloc[i2]),
        "Squeeze_Fire": bool(squeeze_fire.iloc[i2]),
        "i2": i2,
        "Position_Qty": pos_qty,
    }

# ============================================================
# SETUP MODULE 3 — HIGH VOLUME BREAKOUT
# ============================================================

def find_recent_resistance(df: pd.DataFrame, lookback: int = 40):
    if len(df) < lookback + 5:
        return None, None

    highs = df["High"].iloc[-lookback:]
    idx_local = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)].index
    if len(idx_local) == 0:
        return None, None

    i_res = idx_local[-1]
    level = df.loc[i_res, "High"]
    return float(level), i_res


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

    df = compute_indicators(df)

    res_level, res_idx = find_recent_resistance(df, lookback=50)
    if res_level is None:
        return None

    i2 = len(df) - 2
    if i2 <= res_idx:
        return None

    close_i2 = float(df["Close"].iloc[i2])
    vol_i2 = float(df["Volume"].iloc[i2])
    vol_ma = float(df["Volume"].rolling(20).mean().iloc[i2])

    if vol_ma <= 0:
        return None

    vol_ratio_val = vol_i2 / vol_ma

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
    avwap_val = float(df["AVWAP"].iloc[i2])
    if close_i2 < e200 or close_i2 < avwap_val:
        return None

    atr_val = float(df["ATR"].iloc[i2])
    if atr_val <= 0:
        return None

    ei = i2 + 1
    if ei >= len(df):
        return None
    ep = float(df["Open"].iloc[ei])

    q = signal_quality(df, i2)
    if q["total"] < min_q:
        return None

    sl = round(res_level - 0.8 * atr_val, 2)
    tp1 = round(ep + 2.0 * atr_val, 2)
    tp2 = round(ep + 4.0 * atr_val, 2)
    rr = round((tp1 - ep) / (ep - sl), 2) if ep != sl else 0

    risk_amount = total_capital * risk_pct / 100.0
    sl_dist = max(ep - sl, 0.01)
    pos_qty = int(risk_amount / sl_dist)
    pos_qty = max(pos_qty, 0)

    vr_raw = df["Vol_Ratio"].iloc[i2]
    vr = float(vr_raw) if not np.isnan(vr_raw) else round(vol_ratio_val, 2)

    ofi_raw = df["OFI"].iloc[i2]
    ofi_val = float(ofi_raw) if not np.isnan(ofi_raw) else 0.0

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
        "OFI": round(ofi_val, 3),
        "Breakout_Level": round(res_level, 2),
        "i2": i2,
        "Position_Qty": pos_qty,
    }

# ============================================================
# SETUP MODULE 4 — TREND PULLBACK
# ============================================================

def detect_reversal_candle(df: pd.DataFrame, i: int):
    if i <= 0 or i >= len(df):
        return False

    o = df["Open"].iloc[i]
    c = df["Close"].iloc[i]
    h = df["High"].iloc[i]
    l = df["Low"].iloc[i]

    body = abs(c - o)
    lower_wick = o - l if c >= o else c - l
    upper_wick = h - max(o, c)

    if body <= 0:
        return False

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

    df = compute_indicators(df)

    i2 = len(df) - 2
    if i2 < 20:
        return None

    ema21 = df["EMA21"].iloc[i2]
    ema50 = df["EMA50"].iloc[i2]
    ema200 = df["EMA200"].iloc[i2]
    close_i2 = df["Close"].iloc[i2]

    if not (close_i2 > ema21 and close_i2 > ema50 and close_i2 > ema200):
        return None

    recent_lows = df["Low"].iloc[i2 - 5 : i2 + 1]
    near_ema = (abs(recent_lows - ema21) / ema21 < 0.02) | (abs(recent_lows - ema50) / ema50 < 0.02)
    if not near_ema.any():
        return None

    if not detect_reversal_candle(df, i2):
        return None

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

    atr_val = float(df["ATR"].iloc[i2])
    if atr_val <= 0:
        return None

    ei = i2 + 1
    if ei >= len(df):
        return None
    ep = float(df["Open"].iloc[ei])

    q = signal_quality(df, i2)
    if q["total"] < min_q:
        return None

    sl = round(df["Low"].iloc[i2] - 0.5 * atr_val, 2)
    tp1 = round(ep + 2.0 * atr_val, 2)
    tp2 = round(ep + 4.0 * atr_val, 2)
    rr = round((tp1 - ep) / (ep - sl), 2) if ep != sl else 0

    risk_amount = total_capital * risk_pct / 100.0
    sl_dist = max(ep - sl, 0.01)
    pos_qty = int(risk_amount / sl_dist)
    pos_qty = max(pos_qty, 0)

    vr_raw = df["Vol_Ratio"].iloc[i2]
    vr = float(vr_raw) if not np.isnan(vr_raw) else 1.0

    ofi_raw = df["OFI"].iloc[i2]
    ofi_val = float(ofi_raw) if not np.isnan(ofi_raw) else 0.0

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
        "OFI": round(ofi_val, 3),
        "EMA_Touch": "EMA21/50 Pullback",
        "Reversal_Candle": True,
        "i2": i2,
        "Position_Qty": pos_qty,
    }

# ============================================================
# SETUP MODULE 5 — LIQUIDITY SWEEP
# ============================================================

def is_stop_hunt_wick(df: pd.DataFrame, i: int, side: str = "bull"):
    if i <= 1 or i >= len(df):
        return False

    o = df["Open"].iloc[i]
    c = df["Close"].iloc[i]
    h = df["High"].iloc[i]
    l = df["Low"].iloc[i]

    body = abs(c - o)
    if body <= 0:
        return False

    if side == "bull":
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        if lower_wick <= 0:
            return False
        prev_low = df["Low"].iloc[i - 1]
        if (
            lower_wick > body * 1.5
            and lower_wick > upper_wick
            and c > prev_low
        ):
            return True
    else:
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        if upper_wick <= 0:
            return False
        prev_high = df["High"].iloc[i - 1]
        if (
            upper_wick > body * 1.5
            and upper_wick > lower_wick
            and c < prev_high
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

    df = compute_indicators(df)

    i2 = len(df) - 2
    if i2 < 5:
        return None

    if not is_stop_hunt_wick(df, i2, side="bull"):
        return None

    vol = df["Volume"]
    vol_ma = vol.rolling(20).mean()
    if vol_ma.iloc[i2] <= 0:
        return None
    if vol.iloc[i2] < vol_ma.iloc[i2] * 1.5:
        return None

    ofi_raw = df["OFI"].iloc[i2]
    if np.isnan(ofi_raw) or ofi_raw <= 0:
        return None

    if fresh_only and i2 < len(df) - 4:
        return None

    if use_trend:
        tw = get_weekly_trend(ticker)
        if tw is None:
            return None
        if not bool(tw.reindex(df.index, method="ffill").iloc[i2]):
            return None

    atr_val = float(df["ATR"].iloc[i2])
    if atr_val <= 0:
        return None

    ei = i2 + 1
    if ei >= len(df):
        return None
    ep = float(df["Open"].iloc[ei])

    q = signal_quality(df, i2)
    if q["total"] < min_q:
        return None

    sl = round(df["Low"].iloc[i2] - 0.3 * atr_val, 2)
    tp1 = round(ep + 2.5 * atr_val, 2)
    tp2 = round(ep + 5.0 * atr_val, 2)
    rr = round((tp1 - ep) / (ep - sl), 2) if ep != sl else 0

    risk_amount = total_capital * risk_pct / 100.0
    sl_dist = max(ep - sl, 0.01)
    pos_qty = int(risk_amount / sl_dist)
    pos_qty = max(pos_qty, 0)

    vr_raw = df["Vol_Ratio"].iloc[i2]
    vr = float(vr_raw) if not np.isnan(vr_raw) else float(vol.iloc[i2] / max(vol_ma.iloc[i2], 1))

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
        "OFI": round(float(ofi_raw), 3),
        "Sweep_Level": round(float(df["Low"].iloc[i2]), 2),
        "i2": i2,
        "Position_Qty": pos_qty,
    }

# ============================================================
# SETUP MODULE 6 — VCP
# ============================================================

def compute_contraction_series(df: pd.DataFrame, lookback: int = 60):
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

    df = compute_indicators(df)

    contractions = compute_contraction_series(df, lookback=80)
    if len(contractions) < 2:
        return None

    pct_ranges = [c[2] for c in contractions]
    if not (pct_ranges[-1] < pct_ranges[-2]):
        return None

    _, i_pivot, _ = contractions[-1]
    if i_pivot >= len(df) - 3:
        return None

    pivot_high = df["High"].loc[:i_pivot].max()

    vol = df["Volume"]
    vol_pivot_zone = vol.loc[i_pivot - 10 : i_pivot]
    vol_prev_zone = vol.loc[i_pivot - 30 : i_pivot - 11] if i_pivot - 30 >= 0 else None

    if vol_prev_zone is None or len(vol_prev_zone) < 5:
        return None

    if vol_pivot_zone.mean() >= vol_prev_zone.mean():
        return None

    i2 = len(df) - 2
    if i2 <= i_pivot:
        return None

    close_i2 = df["Close"].iloc[i2]
    if close_i2 <= pivot_high:
        return None

    vol_ma = vol.rolling(20).mean()
    if vol_ma.iloc[i2] <= 0:
        return None
    if vol.iloc[i2] < vol_ma.iloc[i2] * 1.5:
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
    avwap_val = float(df["AVWAP"].iloc[i2])
    if close_i2 < e200 or close_i2 < avwap_val:
        return None

    atr_val = float(df["ATR"].iloc[i2])
    if atr_val <= 0:
        return None

    ei = i2 + 1
    if ei >= len(df):
        return None
    ep = float(df["Open"].iloc[ei])

    q = signal_quality(df, i2)
    if q["total"] < min_q:
        return None

    sl = round(pivot_high - 0.8 * atr_val, 2)
    tp1 = round(ep + 2.5 * atr_val, 2)
    tp2 = round(ep + 5.0 * atr_val, 2)
    rr = round((tp1 - ep) / (ep - sl), 2) if ep != sl else 0

    risk_amount = total_capital * risk_pct / 100.0
    sl_dist = max(ep - sl, 0.01)
    pos_qty = int(risk_amount / sl_dist)
    pos_qty = max(pos_qty, 0)

    vr_raw = df["Vol_Ratio"].iloc[i2]
    vr = float(vr_raw) if not np.isnan(vr_raw) else float(vol.iloc[i2] / max(vol_ma.iloc[i2], 1))

    ofi_raw = df["OFI"].iloc[i2]
    ofi_val = float(ofi_raw) if not np.isnan(ofi_raw) else 0.0

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
        "OFI": round(ofi_val, 3),
        "Contraction_Pct": round(contraction_pct, 2),
        "Vol_Dryup": True,
        "i2": i2,
        "Position_Qty": pos_qty,
    }

# ============================================================
# SETUP-SPECIFIC ANALYSIS ENGINE
# ============================================================

def explain_divergence(row, df, i2):
    return f"""
### Divergence Setup Analysis

- Price made a **lower low**, but RSI & MACD showed **higher lows** → bullish divergence.
- Divergence confirmed at index **{i2}** on **{row['Date']}**.
- RSI at signal: **{row.get('RSI', 'N/A')}**.
- Volume ratio: **{row.get('Vol_Ratio', 'N/A')}** (volume confirmation).
- OFI: **{row.get('OFI', 'N/A')}** (order flow shift).
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Quality score: **{row.get('Quality', 'N/A')}**, Grade: **{row.get('Grade', 'N/A')}**.

This is a **high‑probability reversal setup** with strong momentum shift.
"""


def explain_bb_squeeze(row, df, i2):
    sq = "Yes" if row.get("In_Squeeze") else "No"
    fire = "Yes" if row.get("Squeeze_Fire") else "No"

    return f"""
### BB Squeeze Setup Analysis

- Bollinger Bands contracted **inside** Keltner Channels → volatility compression.
- In Squeeze: **{sq}**, Squeeze Fire: **{fire}**.
- Volume ratio: **{row.get('Vol_Ratio', 'N/A')}** → confirms expansion.
- RSI: **{row.get('RSI', 'N/A')}**.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Grade: **{row.get('Grade', 'N/A')}**.

This is a **continuation breakout** after volatility contraction.
"""


def explain_breakout(row, df, i2):
    return f"""
### High Volume Breakout Analysis

- Price broke above resistance at **{row.get('Breakout_Level', 'N/A')}**.
- Volume ratio: **{row.get('Vol_Ratio', 'N/A')}** → strong institutional activity.
- RSI: **{row.get('RSI', 'N/A')}**.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Grade: **{row.get('Grade', 'N/A')}**.

This is a **momentum breakout** with volume confirmation.
"""


def explain_pullback(row, df, i2):
    return f"""
### Trend Pullback (EMA21/50) Analysis

- Price pulled back to **EMA21/50** and formed a bullish reversal candle.
- Reversal candle: **{row.get('Reversal_Candle', True)}**.
- EMA Touch: **{row.get('EMA_Touch', 'EMA21/50')}**.
- Volume contraction on pullback, expansion on reversal.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Grade: **{row.get('Grade', 'N/A')}**.

This is a **trend continuation setup** with clean pullback structure.
"""


def explain_sweep(row, df, i2):
    return f"""
### Liquidity Sweep + Reversal Analysis

- Price swept liquidity below **{row.get('Sweep_Level', 'N/A')}**.
- Long lower wick → stop‑hunt behavior.
- OFI: **{row.get('OFI', 'N/A')}** → order flow flipped positive.
- Volume spike: **{row.get('Vol_Ratio', 'N/A')}**.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Grade: **{row.get('Grade', 'N/A')}**.

This is a **smart‑money reversal setup** with absorption.
"""


def explain_vcp(row, df, i2):
    return f"""
### VCP (Volatility Contraction Pattern) Analysis

- Multiple contractions detected → volatility tightening.
- Last contraction: **{row.get('Contraction_Pct', 'N/A')}%**.
- Volume dry‑up: **{row.get('Vol_Dryup', True)}**.
- Breakout above pivot with volume expansion.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Grade: **{row.get('Grade', 'N/A')}**.

This is a **high‑RR continuation setup** used by Minervini.
"""


def build_setup_analysis(setup_type, row, df, i2):
    if setup_type == "Divergence":
        return explain_divergence(row, df, i2)
    if setup_type == "BB Squeeze":
        return explain_bb_squeeze(row, df, i2)
    if setup_type == "High Volume Breakout":
        return explain_breakout(row, df, i2)
    if setup_type == "Trend Pullback":
        return explain_pullback(row, df, i2)
    if setup_type == "Liquidity Sweep":
        return explain_sweep(row, df, i2)
    if setup_type == "VCP Pattern":
        return explain_vcp(row, df, i2)
    return "No analysis available for this setup."

# ============================================================
# SCREENER UI
# ============================================================

def show_screener():
    st.markdown('<div class="royal-card">', unsafe_allow_html=True)
    st.markdown("### Divergence & Multi-Setup Screener (Live Data)")

    if "selected_setup" not in st.session_state:
        st.session_state["selected_setup"] = "Divergence"

    cols_seg = st.columns(len(SETUP_TYPES))
    for i, setup in enumerate(SETUP_TYPES):
        if cols_seg[i].button(setup, key=f"setup_{setup}"):
            st.session_state["selected_setup"] = setup

    active_setup = st.session_state["selected_setup"]
    seg_html = '<div class="setup-segment">'
    for setup in SETUP_TYPES:
        cls = "active" if setup == active_setup else ""
        seg_html += f'<button class="{cls}">{setup}</button>'
    seg_html += "</div>"
    st.markdown(seg_html, unsafe_allow_html=True)

    st.caption(f"Current setup: **{active_setup}** · Only this setup will be scanned.")

    c1, c2, c3, c4 = st.columns(4)
    universe = c1.selectbox("Universe", ["NIFTY50", "NIFTY200", "Custom"], index=1)
    interval = c2.selectbox("Timeframe", ["1d", "1h", "15m", "1wk"])
    swing_bars = c3.slider("Swing Bars (for swing-based setups)", 3, 10, 5)
    min_q = c4.slider("Min Quality (where applicable)", 0, 100, 50)

    c5, c6, c7 = st.columns(3)
    use_trend = c5.toggle("Weekly Trend Filter", value=True)
    fresh_only = c6.toggle("Fresh Signals Only", value=True)
    sq_only = c7.toggle("Squeeze Signals Only (BB)", value=True)

    if universe == "NIFTY50":
        tickers = NIFTY50
    elif universe == "NIFTY200":
        tickers = NIFTY200
    else:
        tickers = [
            x.strip()
            for x in st.text_input("Custom Tickers (comma separated)", "HAL.NS").split(",")
            if x.strip()
        ]

    total_capital = st.session_state.get("total_capital", 100000.0)
    risk_pct = st.session_state.get("risk_pct", 1.0)

    st.caption(
        f"Position sizing uses total capital = INR {total_capital:,.0f} and risk = {risk_pct:.2f}% per trade."
    )

    run = st.button("Run Screener (Live)")
    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        results = []
        prog = st.progress(0, text=f"Scanning universe for {active_setup} setups...")
        total = len(tickers)
        for idx, t in enumerate(tickers):
            if active_setup == "Divergence":
                r = scan_stock_divergence(
                    t, interval, use_trend, fresh_only,
                    swing_bars, min_q, total_capital, risk_pct
                )
            elif active_setup == "BB Squeeze":
                r = scan_stock_bb_squeeze(
                    t, interval, use_trend, fresh_only,
                    swing_bars, min_q, total_capital, risk_pct,
                    sq_only=sq_only
                )
            elif active_setup == "High Volume Breakout":
                r = scan_stock_breakout(
                    t, interval, use_trend, fresh_only,
                    swing_bars, min_q, total_capital, risk_pct
                )
            elif active_setup == "Trend Pullback":
                r = scan_stock_pullback(
                    t, interval, use_trend, fresh_only,
                    swing_bars, min_q, total_capital, risk_pct
                )
            elif active_setup == "Liquidity Sweep":
                r = scan_stock_sweep(
                    t, interval, use_trend, fresh_only,
                    swing_bars, min_q, total_capital, risk_pct
                )
            elif active_setup == "VCP Pattern":
                r = scan_stock_vcp(
                    t, interval, use_trend, fresh_only,
                    swing_bars, min_q, total_capital, risk_pct
                )
            else:
                r = None

            if r:
                results.append(r)
            prog.progress((idx + 1) / total, text=f"Scanning {t}")
        prog.empty()

        if results:
            df_res = pd.DataFrame(results)
            df_res = df_res.replace([np.inf, -np.inf], np.nan).dropna(how="all")

            desired_cols = get_screener_columns_for_setup(active_setup)
            cols_present = [c for c in desired_cols if c in df_res.columns]
            if cols_present:
                df_res = df_res[cols_present]

            sort_cols = [c for c in ["Grade", "Quality"] if c in df_res.columns]
            if sort_cols:
                df_res = df_res.sort_values(sort_cols, ascending=[False] * len(sort_cols))

            st.session_state["sr"] = df_res
            st.success(f"Found {len(df_res)} setups for {active_setup}")
        else:
            st.session_state.pop("sr", None)
            st.info(f"No qualifying {active_setup} setups found with current filters.")

    if "sr" in st.session_state:
        df_res = st.session_state["sr"]
        st.markdown('<div class="royal-card">', unsafe_allow_html=True)
        st.markdown("### Screener Results")
        st.dataframe(df_res, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="royal-card">', unsafe_allow_html=True)
        st.markdown("### Stock Analysis")

        if "Ticker" in df_res.columns:
            tickers_list = df_res["Ticker"].tolist()
        elif "Symbol" in df_res.columns:
            tickers_list = df_res["Symbol"].tolist()
        else:
            tickers_list = []

        if tickers_list:
            selected = st.selectbox("Select a stock for detailed analysis", tickers_list)
            if "Ticker" in df_res.columns:
                row = df_res[df_res["Ticker"] == selected].iloc[0]
                ticker_for_analysis = row["Ticker"]
            else:
                row = df_res[df_res["Symbol"] == selected].iloc[0]
                ticker_for_analysis = row["Symbol"]

            df_full = compute_indicators(load_data(ticker_for_analysis, interval))

            if "i2" in row:
                i2 = int(row["i2"])
            else:
                i2 = len(df_full) - 1

            analysis_text = build_setup_analysis(
                active_setup, row, df_full, i2
            )

            st.markdown(
                f"**Why this stock is on the radar ({row.get('Symbol', ticker_for_analysis)} / {ticker_for_analysis}):**"
            )
            st.markdown(analysis_text)
        else:
            st.info("No rows available for analysis.")
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# MAIN
# ============================================================

def main():
    apply_theme()
    st.set_page_config(page_title="Sniper Terminal v3", layout="wide")
    st.title("Sniper Terminal v3 — Multi-Setup Screener")

    with st.sidebar:
        st.markdown("### Risk & Capital")
        total_capital = st.number_input("Total Capital (INR)", min_value=10000.0, value=100000.0, step=5000.0)
        risk_pct = st.slider("Risk % per Trade", 0.1, 5.0, 1.0, 0.1)
        st.session_state["total_capital"] = total_capital
        st.session_state["risk_pct"] = risk_pct

    show_screener()


if __name__ == "__main__":
    main()
```", 0.1, 5.0, 1.0, 0.1)
        st.session_state["total_capital"] = total_capital
        st.session_state["risk_pct"] = risk_pct

    show_screener()


if __name__ == "__main__":
    main()
