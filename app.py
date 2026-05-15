import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nifty50 import NIFTY50
from nifty200 import NIFTY200
from nifty500 import NIFTY500


# ============================================================
# THEME & GLOBAL CONFIG
# ============================================================

def apply_theme():
    st.set_page_config(
        page_title="Sniper Terminal v4 — Royal Gold",
        layout="wide",
    )
    if "plot_theme" not in st.session_state:
        st.session_state["plot_theme"] = "plotly_dark"

    st.markdown(
        """
        <style>
        .royal-title {
            font-size: 32px;
            font-weight: 800;
            color: #f5c542;
        }
        .royal-subtitle {
            font-size: 16px;
            font-weight: 500;
            color: #d0d0d0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATA LOADER WITH RETRY & SAFETY
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(ticker: str, interval: str = "1d", years: int = 3) -> pd.DataFrame:
    for _ in range(3):
        try:
            df = yf.download(
                ticker,
                period=f"{years}y",
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if df is not None and not df.empty:
                df = df.dropna()
                return df
        except Exception:
            time.sleep(1)
    return pd.DataFrame()


# ============================================================
# CORE INDICATORS ENGINE
# ============================================================

def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / length, adjust=False).mean()
    return atr


def compute_ofi(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    vol = df["Volume"]
    prev_close = close.shift(1)

    direction = np.sign(close - prev_close).fillna(0)
    ofi = direction * vol
    return ofi.rolling(5).mean().fillna(0)


def compute_avwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vol = df["Volume"].cumsum()
    cum_pv = (tp * df["Volume"]).cumsum()
    avwap = cum_pv / cum_vol.replace(0, np.nan)
    return avwap


@st.cache_data(show_spinner=False)
def compute_indicators(df: pd.DataFrame, swing_bars: int = 5) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # EMAs
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # RSI & ATR
    df["RSI"] = compute_rsi(df["Close"], length=14)
    df["ATR"] = compute_atr(df, length=14)

    # Volume metrics
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    df["Vol_Ratio"] = (df["Volume"] / df["Vol_MA20"].replace(0, np.nan)).fillna(0)

    # Order flow
    df["OFI"] = compute_ofi(df)

    # AVWAP
    df["AVWAP"] = compute_avwap(df)

    # Bollinger Bands
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Mid"] = mid
    df["BB_Upper"] = mid + 2 * std
    df["BB_Lower"] = mid - 2 * std

    # Keltner Channels
    atr20 = df["ATR"].rolling(20).mean()
    df["KC_Mid"] = mid
    df["KC_Upper"] = mid + 1.5 * atr20
    df["KC_Lower"] = mid - 1.5 * atr20

    # Squeeze condition
    df["In_Squeeze"] = (df["BB_Upper"] < df["KC_Upper"]) & (df["BB_Lower"] > df["KC_Lower"])
    df["Squeeze_Fire"] = df["In_Squeeze"].shift(1) & (~df["In_Squeeze"])

    # Simple divergence flags (price vs RSI)
    df["Bull_Div"] = False
    df["Bear_Div"] = False
    for i in range(swing_bars, len(df) - swing_bars):
        # local lows
        if (
            df["Low"].iloc[i] < df["Low"].iloc[i - 1]
            and df["Low"].iloc[i] < df["Low"].iloc[i + 1]
        ):
            prev_idx = i - swing_bars
            if prev_idx >= 0:
                if (
                    df["Low"].iloc[i] < df["Low"].iloc[prev_idx]
                    and df["RSI"].iloc[i] > df["RSI"].iloc[prev_idx]
                ):
                    df.at[df.index[i], "Bull_Div"] = True

        # local highs
        if (
            df["High"].iloc[i] > df["High"].iloc[i - 1]
            and df["High"].iloc[i] > df["High"].iloc[i + 1]
        ):
            prev_idx = i - swing_bars
            if prev_idx >= 0:
                if (
                    df["High"].iloc[i] > df["High"].iloc[prev_idx]
                    and df["RSI"].iloc[i] < df["RSI"].iloc[prev_idx]
                ):
                    df.at[df.index[i], "Bear_Div"] = True

    return df


# ============================================================
# WEEKLY TREND & SIGNAL QUALITY ENGINE
# ============================================================

@st.cache_data(show_spinner=False)
def get_weekly_trend(ticker: str):
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
    score = 0
    rsi_val = float(df["RSI"].iloc[i])
    vr = float(df["Vol_Ratio"].iloc[i])
    atr_val = float(df["ATR"].iloc[i])

    if 40 <= rsi_val <= 70:
        score += 25

    if vr > 1.2:
        score += 25

    atr_ma50 = df["ATR"].rolling(50).mean().iloc[i]
    if not np.isnan(atr_ma50) and atr_val > atr_ma50:
        score += 25

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
    if df.empty or len(df) < 120:
        return None

    df = compute_indicators(df, swing_bars=swing_bars)

    i2 = len(df) - 2
    if i2 < swing_bars + 2:
        return None

    if not (df["Bull_Div"].iloc[i2] or df["Bear_Div"].iloc[i2]):
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

    if df["Bull_Div"].iloc[i2]:
        sl = round(float(df["Low"].iloc[i2]) - 0.5 * atr, 2)
        tp1 = round(ep + 2.0 * atr, 2)
        tp2 = round(ep + 4.0 * atr, 2)
    else:
        sl = round(float(df["High"].iloc[i2]) + 0.5 * atr, 2)
        tp1 = round(ep - 2.0 * atr, 2)
        tp2 = round(ep - 4.0 * atr, 2)

    rr = round((tp1 - ep) / (ep - sl), 2) if ep != sl else 0

    risk_amount = total_capital * risk_pct / 100.0
    sl_dist = max(abs(ep - sl), 0.01)
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
        "i2": i2,
        "Position_Qty": qty,
    }


# ============================================================
# SETUP ENGINE 2 — BB SQUEEZE
# ============================================================

def scan_stock_bb_squeeze(
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
    if i2 < 25:
        return None

    if not bool(df["Squeeze_Fire"].iloc[i2]):
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
        "In_Squeeze": bool(df["In_Squeeze"].iloc[i2]),
        "Squeeze_Fire": bool(df["Squeeze_Fire"].iloc[i2]),
        "Vol_Ratio": round(vr, 2),
        "RSI": round(float(df["RSI"].iloc[i2]), 1),
        "i2": i2,
        "Position_Qty": qty,
    }


# ============================================================
# SETUP ENGINE 3 — HIGH VOLUME BREAKOUT
# ============================================================

def find_recent_resistance(df: pd.DataFrame, lookback: int = 40):
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

    if not (close_i2 > ema21 and close_i2 > ema50 and close_i2 > ema200):
        return None

    recent_lows = df["Low"].iloc[i2 - 5 : i2 + 1]
    near_ema = (
        (abs(recent_lows - ema21) / ema21 < 0.02)
        | (abs(recent_lows - ema50) / ema50 < 0.02)
    )
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
# SETUP ENGINE 5 — LIQUIDITY SWEEP
# ============================================================

def is_stop_hunt_wick(df: pd.DataFrame, i: int, side: str = "bull"):
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
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        prev_low = float(df["Low"].iloc[i - 1])

        if (
            lower_wick > body * 1.5
            and lower_wick > upper_wick
            and c > prev_low
        ):
            return True
    else:
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        prev_high = float(df["High"].iloc[i - 1])

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

    df = compute_indicators(df, swing_bars=swing_bars)

    i2 = len(df) - 2
    if i2 < 5:
        return None

    if not is_stop_hunt_wick(df, i2, side="bull"):
        return None

    vol = df["Volume"]
    vol_ma20 = vol.rolling(20).mean()
    if vol_ma20.iloc[i2] <= 0:
        return None
    if vol.iloc[i2] < vol_ma20.iloc[i2] * 1.5:
        return None

    ofi = float(df["OFI"].iloc[i2])
    if np.isnan(ofi) or ofi <= 0:
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
# SETUP ENGINE 6 — VCP
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

    df = compute_indicators(df, swing_bars=swing_bars)

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
# SMART COLUMNS & EXPLANATIONS
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


def explain_divergence(row, df, i2):
    return f"""
### Divergence Setup Analysis

- Price made a **lower low**, but RSI showed **higher lows** → bullish divergence.
- Divergence confirmed at index **{i2}** on **{row['Date']}**.
- RSI: **{row.get('RSI')}**, Volume Ratio: **{row.get('Vol_Ratio')}**, OFI: **{row.get('OFI')}**.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Quality Score: **{row.get('Quality')}**, Grade: **{row.get('Grade')}**.
"""


def explain_bb_squeeze(row, df, i2):
    return f"""
### BB Squeeze Setup Analysis

- Bollinger Bands contracted **inside** Keltner Channels → volatility compression.
- In Squeeze: **{row.get('In_Squeeze')}**, Squeeze Fire: **{row.get('Squeeze_Fire')}**.
- Volume Ratio: **{row.get('Vol_Ratio')}**, RSI: **{row.get('RSI')}**.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Grade: **{row.get('Grade')}**.
"""


def explain_breakout(row, df, i2):
    return f"""
### High Volume Breakout Analysis

- Price broke above resistance: **{row.get('Breakout_Level')}**.
- Volume Ratio: **{row.get('Vol_Ratio')}** → confirms strong participation.
- RSI: **{row.get('RSI')}**, OFI: **{row.get('OFI')}**.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Grade: **{row.get('Grade')}**.
"""


def explain_pullback(row, df, i2):
    return f"""
### Trend Pullback Analysis

- Price pulled back to **EMA21/50** and formed a bullish reversal candle.
- Volume on reversal > 1.2× pullback volume.
- RSI: **{row.get('RSI')}**, Volume Ratio: **{row.get('Vol_Ratio')}**.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Grade: **{row.get('Grade')}**.
"""


def explain_sweep(row, df, i2):
    return f"""
### Liquidity Sweep Analysis

- Price swept liquidity below **{row.get('Sweep_Level')}** and reversed strongly.
- OFI positive → buyers stepped in aggressively.
- Volume Ratio: **{row.get('Vol_Ratio')}**, RSI: **{row.get('RSI')}**.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Grade: **{row.get('Grade')}**.
"""


def explain_vcp(row, df, i2):
    return f"""
### VCP Pattern Analysis

- Volatility contraction detected: **{row.get('Contraction_Pct')}%**.
- Volume dry-up confirmed near pivot.
- Breakout above pivot high with strong volume.
- Entry: **{row['Entry']}**, SL: **{row['SL']}**, RR: **{row['R_R']}**.
- Grade: **{row.get('Grade')}**.
"""


# ============================================================
# SETUP DISPATCHER & SCREENER
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
) -> pd.DataFrame:
    rows = []
    for ticker in universe:
        try:
            sig = run_setup_scan(
                setup_type,
                ticker,
                interval,
                use_trend,
                fresh_only,
                swing_bars,
                min_q,
                total_capital,
                risk_pct,
            )
            if sig:
                rows.append(sig)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    sort_col = "Quality" if "Quality" in df.columns else "Grade"
    df = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    return df


# ============================================================
# BACKTEST ENGINE
# ============================================================

def simulate_trade(entry, sl, tp1, tp2, df_slice):
    for i in range(len(df_slice)):
        low = df_slice["Low"].iloc[i]
        high = df_slice["High"].iloc[i]

        if low <= sl:
            return "SL", sl, (sl - entry)

        if high >= tp1:
            if high >= tp2:
                return "TP2", tp2, (tp2 - entry)
            return "TP1", tp1, (tp1 - entry)

    last_close = df_slice["Close"].iloc[-1]
    return "EXIT", last_close, (last_close - entry)


def backtest_setup(df, signals, risk_pct, total_capital):
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

        df_slice = df.iloc[i2 + 1 : i2 + 40]
        if df_slice.empty:
            continue

        outcome, exit_price, pnl = simulate_trade(entry, sl, tp1, tp2, df_slice)

        risk_amount = capital * risk_pct / 100.0
        sl_dist = max(abs(entry - sl), 0.01)
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


def autotrade_signal_executor(df, sig, risk_pct, total_capital):
    i2 = sig["i2"]
    if i2 + 1 >= len(df):
        return {"status": "no_data"}

    entry = sig["Entry"]
    sl = sig["SL"]
    tp1 = sig["TP1"]
    tp2 = sig["TP2"]

    df_slice = df.iloc[i2 + 1 : i2 + 40]
    if df_slice.empty:
        return {"status": "no_data"}

    outcome, exit_price, pnl = simulate_trade(entry, sl, tp1, tp2, df_slice)

    risk_amount = total_capital * risk_pct / 100.0
    sl_dist = max(abs(entry - sl), 0.01)
    qty = max(int(risk_amount / sl_dist), 1)
    profit = qty * pnl

    return {
        "status": "executed",
        "Outcome": outcome,
        "Entry": entry,
        "Exit": exit_price,
        "Qty": qty,
        "PnL": round(profit, 2),
    }


# ============================================================
# VOLUME PROFILE ENGINE
# ============================================================

def compute_volume_profile(df: pd.DataFrame, bins: int = 40):
    if df is None or df.empty:
        return np.array([]), np.array([]), None

    prices = df["Close"].astype(float)
    vols = df["Volume"].astype(float)

    if prices.isna().all() or vols.isna().all():
        return np.array([]), np.array([]), None

    try:
        hist, bin_edges = np.histogram(prices, bins=bins, weights=vols)
    except ValueError:
        return np.array([]), np.array([]), None

    if hist.max() <= 0:
        return bin_edges, hist, None

    poc_idx = np.argmax(hist)
    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0
    return bin_edges, hist, poc_price


# ============================================================
# CHART ENGINE
# ============================================================

def plot_chart(df, ticker=None):
    if df is None or df.empty:
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
    for col, color in [
        ("EMA21", "cyan"),
        ("EMA50", "orange"),
        ("EMA200", "yellow"),
    ]:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=col,
                    line=dict(color=color, width=1),
                ),
                row=1, col=1
            )

    # AVWAP
    if "AVWAP" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["AVWAP"],
                mode="lines",
                name="AVWAP",
                line=dict(color="magenta", width=1),
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
                marker=dict(color="lime", size=10, symbol="triangle-up"),
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
                marker=dict(color="red", size=10, symbol="triangle-down"),
                name="Bear Div",
            ),
            row=1, col=1
        )

    # Volume
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color="rgba(100,149,237,0.7)",
        ),
        row=2, col=1
    )

    # RSI
    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="white", width=1),
            ),
            row=2, col=1
        )

    title = f"{ticker}" if ticker else "Chart"
    fig.update_layout(
        template=st.session_state.get("plot_theme", "plotly_dark"),
        height=700,
        margin=dict(l=10, r=10, t=30, b=20),
        xaxis_rangeslider_visible=False,
        title=title,
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

    col0, col1, col2 = st.columns(3)
    with col0:
        universe_choice = st.selectbox("Universe", ["NIFTY50", "NIFTY200", "NIFTY500"])
    if universe_choice == "NIFTY50":
        universe = NIFTY50
    elif universe_choice == "NIFTY200":
        universe = NIFTY200
    else:
        universe = NIFTY500

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

        sig = run_setup_scan(
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

        if not sig:
            st.warning("No signals found for backtest.")
            return

        signals = [sig]
        bt = backtest_setup(df, signals, risk_pct, total_capital)
        if bt.empty:
            st.warning("No trades generated.")
            return

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

        if poc is None or len(vol) == 0:
            st.warning("Not enough data to compute volume profile.")
            return

        st.write("POC:", round(poc, 2))
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
