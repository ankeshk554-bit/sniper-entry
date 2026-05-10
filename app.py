import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.linear_model import LogisticRegression

from shoonya_api import ShoonyaAPI  # your separate file


# =========================
# SESSION STATE INITIALIZE
# =========================

def init_state():
    if "shoonya" not in st.session_state:
        st.session_state["shoonya"] = ShoonyaAPI()

    if "at_pos" not in st.session_state:
        st.session_state["at_pos"] = {}  # {sym: {...}}

    if "at_log" not in st.session_state:
        st.session_state["at_log"] = []

    if "at_pnl" not in st.session_state:
        st.session_state["at_pnl"] = 0.0

    if "algo_running" not in st.session_state:
        st.session_state["algo_running"] = False

    if "algo_last_entry" not in st.session_state:
        st.session_state["algo_last_entry"] = {}  # {sym: ts}

    if "ltp" not in st.session_state:
        st.session_state["ltp"] = {}  # {sym: price}

    if "token_map" not in st.session_state:
        st.session_state["token_map"] = {}  # token -> sym

    if "symbol_map" not in st.session_state:
        st.session_state["symbol_map"] = {}  # sym -> token

    if "at_mode" not in st.session_state:
        st.session_state["at_mode"] = "paper"

    if "risk_per_trade" not in st.session_state:
        st.session_state["risk_per_trade"] = 2000.0

    if "max_positions" not in st.session_state:
        st.session_state["max_positions"] = 5

    if "daily_loss_limit" not in st.session_state:
        st.session_state["daily_loss_limit"] = -10000.0

    if "daily_pnl" not in st.session_state:
        st.session_state["daily_pnl"] = 0.0

    if "daily_date" not in st.session_state:
        st.session_state["daily_date"] = str(pd.Timestamp.today().date())

    if "daily_profit_target" not in st.session_state:
        st.session_state["daily_profit_target"] = 10000.0

    if "symbol_trades_today" not in st.session_state:
        st.session_state["symbol_trades_today"] = {}

    if "max_trades_per_symbol" not in st.session_state:
        st.session_state["max_trades_per_symbol"] = 3

    if "trade_history" not in st.session_state:
        st.session_state["trade_history"] = []

    if "equity_curve" not in st.session_state:
        st.session_state["equity_curve"] = []

    if "max_equity" not in st.session_state:
        st.session_state["max_equity"] = 0.0

    if "vol_regime" not in st.session_state:
        st.session_state["vol_regime"] = "NORMAL"

    if "auto_pause" not in st.session_state:
        st.session_state["auto_pause"] = False

    if "ml_model" not in st.session_state:
        st.session_state["ml_model"] = LogisticRegression()
        st.session_state["ml_trained"] = False


init_state()


# =========================
# UNIVERSE / HELPERS
# =========================

NIFTY50 = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
    "ITC.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "HINDUNILVR.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "HCLTECH.NS",
    "POWERGRID.NS", "NTPC.NS", "ONGC.NS", "TATAMOTORS.NS", "M&M.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "NESTLEIND.NS", "DRREDDY.NS",
    "CIPLA.NS", "GRASIM.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "HDFCLIFE.NS", "SBILIFE.NS", "BRITANNIA.NS",
    "HEROMOTOCO.NS", "EICHERMOT.NS", "COALINDIA.NS", "BPCL.NS",
    "IOC.NS", "SHREECEM.NS", "BAJAJ-AUTO.NS", "HINDALCO.NS",
    "DIVISLAB.NS", "TECHM.NS", "UPL.NS", "TATACONSUM.NS",
]


def load_data(symbol: str, tf: str, years: int = 1) -> pd.DataFrame:
    # Placeholder: replace with your real data loader (yfinance, broker, etc.)
    # Must return OHLCV DataFrame with DatetimeIndex
    return pd.DataFrame()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Simple AVWAP from first bar
    cum_vol = df["Volume"].cumsum()
    cum_vp = (df["Close"] * df["Volume"]).cumsum()
    df["AVWAP"] = cum_vp / (cum_vol + 1e-9)

    # Squeeze (placeholder)
    df["Squeeze"] = False
    df["Squeeze_Fire"] = False

    # VCP flag (placeholder)
    df["VCP_Flag"] = False

    return df


def compute_divergences(df: pd.DataFrame, bars: int = 5):
    # Placeholder divergence logic
    return [], []


def signal_quality(df: pd.DataFrame, idx: int):
    # Placeholder quality scoring
    return {"total": 80}


# =========================
# BREAKOUT / STRENGTH / LIVE SCORING
# =========================

def compute_breakout_status(sym, ltp, df):
    out = {"AVWAP": 0, "EMA21": 0, "EMA50": 0, "PDH": 0}

    avwap = df["AVWAP"].iloc[-1]
    e21 = df["EMA21"].iloc[-1]
    e50 = df["EMA50"].iloc[-1]

    try:
        df_d = load_data(sym, "1d", years=1)
        pdh = df_d["High"].iloc[-2] if len(df_d) > 1 else df_d["High"].iloc[-1]
    except Exception:
        pdh = df["High"].iloc[-2] if len(df) > 1 else df["High"].iloc[-1]

    if ltp > avwap:
        out["AVWAP"] = 1
    if ltp > e21:
        out["EMA21"] = 1
    if ltp > e50:
        out["EMA50"] = 1
    if ltp > pdh:
        out["PDH"] = 1

    return out


def compute_multi_breakout_score(sym, ltp, df):
    score = 0

    avwap = df["AVWAP"].iloc[-1]
    e21 = df["EMA21"].iloc[-1]
    e50 = df["EMA50"].iloc[-1]
    e200 = df["EMA200"].iloc[-1]
    last_high = df["High"].iloc[-1]

    try:
        df_d = load_data(sym, "1d", years=1)
        pdh = df_d["High"].iloc[-2] if len(df_d) > 1 else df_d["High"].iloc[-1]
    except Exception:
        pdh = df["High"].iloc[-2] if len(df) > 1 else df["High"].iloc[-1]

    if ltp > avwap:
        score += 25
    if ltp > e21:
        score += 25
    if ltp > e50:
        score += 25
    if ltp > pdh:
        score += 25

    if ltp > last_high:
        score += 10

    if ltp > e200:
        score += 10

    return min(score, 100)


def live_signal_score(sym, ltp, df):
    score = 0

    if ltp > df["EMA200"].iloc[-1]:
        score += 15
    if ltp > df["AVWAP"].iloc[-1]:
        score += 10

    rsi = df["RSI"].iloc[-1]
    if 30 < rsi < 50:
        score += 10

    if df["Squeeze"].iloc[-1]:
        score += 20
    if df["Squeeze_Fire"].iloc[-1]:
        score += 30

    if "VCP_Flag" in df.columns and df["VCP_Flag"].iloc[-1]:
        score += 25

    bull, _ = compute_divergences(df, bars=5)
    if bull:
        score += 30

    return min(score, 100)


# =========================
# VOLATILITY / BREADTH / EQUITY
# =========================

def compute_vol_regime():
    try:
        df = load_data("^NSEI", "1d", years=1)
        df = compute_indicators(df)
        atr = df["ATR"].iloc[-1]
        close = df["Close"].iloc[-1]
        atr_pct = (atr / close) * 100

        if atr_pct < 0.8:
            regime = "LOW"
        elif atr_pct < 1.5:
            regime = "NORMAL"
        else:
            regime = "HIGH"
    except Exception:
        regime = "NORMAL"

    st.session_state["vol_regime"] = regime
    return regime


def compute_market_breadth():
    adv = dec = total = 0

    for sym in NIFTY50:
        try:
            df = load_data(sym, "1d", years=1)
            df = compute_indicators(df)
            close = df["Close"].iloc[-1]
            e21 = df["EMA21"].iloc[-1]
            total += 1
            if close > e21:
                adv += 1
            else:
                dec += 1
        except Exception:
            continue

    if total == 0:
        return {"adv": 0, "dec": 0, "breadth": 0.0}

    breadth = (adv - dec) / total * 100
    return {"adv": adv, "dec": dec, "breadth": breadth}


def update_equity_curve():
    eq = st.session_state.get("at_pnl", 0.0)
    st.session_state["max_equity"] = max(st.session_state["max_equity"], eq)
    st.session_state["equity_curve"].append(
        {"ts": pd.Timestamp.now(), "equity": eq}
    )


def compute_drawdown():
    eq = st.session_state.get("at_pnl", 0.0)
    max_eq = st.session_state.get("max_equity", 0.0)
    if max_eq <= 0:
        return 0.0
    return (eq - max_eq) / max_eq * 100


# =========================
# PYRAMIDING / TRAILING SL / SECTOR
# =========================

SECTORS = {
    "BANKS": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
    "PHARMA": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS"],
    "AUTO": ["TATAMOTORS.NS", "M&M.NS", "MARUTI.NS"],
}


def try_pyramid(sym, ltp, df, strength):
    pos = st.session_state["at_pos"].get(sym)
    if not pos or pos["status"] != "OPEN":
        return

    entry = pos["entry"]
    qty = pos["qty"]

    if ltp <= entry:
        return

    pyramids = pos.get("pyramids", 0)
    if pyramids >= 3:
        return

    if strength < 80:
        return

    avwap = df["AVWAP"].iloc[-1]
    e21 = df["EMA21"].iloc[-1]
    e50 = df["EMA50"].iloc[-1]

    if not (ltp > avwap and ltp > e21 and ltp > e50):
        return

    add_qty = max(1, qty // 2)

    if st.session_state["at_mode"] == "shoonya":
        api = st.session_state["shoonya"]
        tsym = sym.replace(".NS", "-EQ")
        api.place_order(
            buy_or_sell="B",
            product_type="I",
            exchange="NSE",
            tradingsymbol=tsym,
            quantity=add_qty,
            price_type="MKT",
            remarks="SniperAlgoPyramid",
        )

    pos["qty"] += add_qty
    pos["pyramids"] = pyramids + 1

    st.session_state["at_log"].append(
        f"[PYRAMID] {sym} +{add_qty} @ {ltp:.2f} | Strength {strength}"
    )


def update_trailing_sl(sym, ltp, df):
    pos = st.session_state["at_pos"].get(sym)
    if not pos or pos["status"] != "OPEN":
        return

    atr = df["ATR"].iloc[-1]
    new_sl = ltp - 2 * atr

    if new_sl > pos["sl"]:
        pos["sl"] = new_sl
        st.session_state["at_log"].append(
            f"[TRAIL SL] {sym} → {new_sl:.2f}"
        )


def compute_sector_strength():
    sector_strength = {}

    for sector, symbols in SECTORS.items():
        scores = []
        for sym in symbols:
            ltp = st.session_state["ltp"].get(sym)
            if not ltp:
                continue
            try:
                df = load_data(sym, "15m", years=1)
                df = compute_indicators(df)
                strength = compute_multi_breakout_score(sym, ltp, df)
                scores.append(strength)
            except Exception:
                continue

        sector_strength[sector] = sum(scores) / len(scores) if scores else 0.0

    return sector_strength


# =========================
# ML FILTER
# =========================

def ml_features(df, strength, has_div):
    return np.array([
        df["ATR"].iloc[-1],
        df["RSI"].iloc[-1],
        strength,
        1 if df["Close"].iloc[-1] > df["EMA200"].iloc[-1] else 0,
        1 if df["Squeeze"].iloc[-1] else 0,
        1 if df["VCP_Flag"].iloc[-1] else 0,
        1 if has_div else 0,
    ]).reshape(1, -1)


def ml_predict(df, strength, has_div):
    if not st.session_state["ml_trained"]:
        return 1
    X = ml_features(df, strength, has_div)
    return st.session_state["ml_model"].predict(X)[0]


# =========================
# AUTO-ENTRY DECISION
# =========================

def should_auto_enter(sym, df, ltp, strength_score):
    if strength_score < 70:
        return False, "Strength < 70"

    try:
        bull, _ = compute_divergences(df, bars=5)
        if not bull:
            return False, "No bullish divergence"

        i1, i2 = bull[-1]
        if i2 < len(df) - 3:
            return False, "Divergence not fresh"

        q = signal_quality(df, i2)
        if q["total"] < 60:
            return False, "Signal quality < 60"

        avwap = df["AVWAP"].iloc[i2]
        e200 = df["EMA200"].iloc[i2]
        if ltp < avwap or ltp < e200:
            return False, "Price below AVWAP/EMA200"

        return True, {"i2": i2, "q": q}

    except Exception as e:
        return False, f"Error: {e}"


# =========================
# AUTO-ENTRY ENGINE
# =========================

def auto_entry_engine():
    if not st.session_state.get("algo_running", False):
        return

    today = str(pd.Timestamp.today().date())
    if st.session_state["daily_date"] != today:
        st.session_state["daily_date"] = today
        st.session_state["daily_pnl"] = 0.0
        st.session_state["symbol_trades_today"] = {}

    if st.session_state["daily_pnl"] <= st.session_state["daily_loss_limit"]:
        if st.session_state["algo_running"]:
            st.session_state["algo_running"] = False
            st.session_state["at_log"].append(
                f"[CIRCUIT BREAKER] Daily loss limit hit ({st.session_state['daily_pnl']:.2f}). Algo stopped."
            )
        return

    if st.session_state["daily_pnl"] >= st.session_state["daily_profit_target"]:
        if st.session_state["algo_running"]:
            st.session_state["algo_running"] = False
            st.session_state["at_log"].append(
                f"[CIRCUIT BREAKER] Daily profit target hit ({st.session_state['daily_pnl']:.2f}). Algo stopped."
            )
        return

    regime = st.session_state.get("vol_regime", "NORMAL")
    if regime == "HIGH":
        st.session_state["auto_pause"] = True
        st.session_state["at_log"].append(
            "[AUTO-PAUSE] Volatility regime = HIGH. Auto-entries paused."
        )

    if st.session_state.get("auto_pause", False):
        return

    open_positions = sum(1 for p in st.session_state["at_pos"].values() if p["status"] == "OPEN")
    if open_positions >= st.session_state["max_positions"]:
        return

    symbols = NIFTY50[:50]
    mode = st.session_state.get("at_mode", "paper")
    risk_per_trade = st.session_state["risk_per_trade"]

    for sym in symbols:
        if sym in st.session_state["at_pos"] and st.session_state["at_pos"][sym]["status"] == "OPEN":
            continue

        open_positions = sum(1 for p in st.session_state["at_pos"].values() if p["status"] == "OPEN")
        if open_positions >= st.session_state["max_positions"]:
            break

        ltp = st.session_state["ltp"].get(sym)
        if not ltp:
            continue

        try:
            df = load_data(sym, "15m", years=1)
            if df.empty:
                continue

            df = compute_indicators(df)
            strength = compute_multi_breakout_score(sym, ltp, df)

            ok, info = should_auto_enter(sym, df, ltp, strength)
            if not ok:
                continue

            i2 = info["i2"]
            atr = df["ATR"].iloc[i2]
            if atr <= 0:
                continue

            sl = ltp - 1.5 * atr
            tp1 = ltp + 2.5 * atr
            risk_per_share = max(ltp - sl, 0.1)
            qty = max(1, int(risk_per_trade / risk_per_share))

            sym_trades = st.session_state["symbol_trades_today"].get(sym, 0)
            if sym_trades >= st.session_state["max_trades_per_symbol"]:
                continue

            last = st.session_state["algo_last_entry"].get(sym, 0)
            if time.time() - last < 1200:
                continue

            ml_ok = ml_predict(df, strength, has_div=True)
            if ml_ok != 1:
                continue

            if mode == "shoonya":
                api = st.session_state["shoonya"]
                tsym = sym.replace(".NS", "-EQ")
                api.place_order(
                    buy_or_sell="B",
                    product_type="I",
                    exchange="NSE",
                    tradingsymbol=tsym,
                    quantity=qty,
                    price_type="MKT",
                    remarks="SniperAlgoAutoEntry",
                )

            st.session_state["at_pos"][sym] = {
                "dir": "BUY",
                "entry": ltp,
                "sl": sl,
                "tp1": tp1,
                "qty": qty,
                "status": "OPEN",
            }

            st.session_state["algo_last_entry"][sym] = time.time()
            st.session_state["symbol_trades_today"][sym] = sym_trades + 1

            st.session_state["at_log"].append(
                f"[AUTO-ENTRY] {sym} BUY {qty} @ {ltp:.2f} | SL {sl:.2f} | TP {tp1:.2f} | Strength {strength}"
            )

            st.session_state["trade_history"].append({
                "Timestamp": str(pd.Timestamp.now()),
                "Symbol": sym,
                "Side": "BUY",
                "Entry": ltp,
                "SL": sl,
                "TP1": tp1,
                "Qty": qty,
                "Reason": "AUTO-ENTRY",
            })

        except Exception as e:
            st.session_state["at_log"].append(f"[AUTO-ENTRY ERROR] {sym}: {e}")
            continue


# =========================
# WEBSOCKET TICK HANDLER
# =========================

def handle_shoonya_tick(msg: dict):
    try:
        if msg.get("t") != "tk":
            return

        token = msg.get("tk")
        lp = float(msg.get("lp", 0))

        token_map = st.session_state["token_map"]
        sym = token_map.get(token)
        if not sym:
            return

        st.session_state["ltp"][sym] = lp

        today = str(pd.Timestamp.today().date())
        if st.session_state["daily_date"] != today:
            st.session_state["daily_date"] = today
            st.session_state["daily_pnl"] = 0.0
            st.session_state["symbol_trades_today"] = {}

        pos = st.session_state["at_pos"]
        if sym in pos:
            p = pos[sym]
            if p["status"] == "OPEN":
                entry = p["entry"]
                sl = p["sl"]
                tp1 = p["tp1"]
                qty = p["qty"]

                if p["dir"] == "BUY":
                    if lp <= sl:
                        pnl = (sl - entry) * qty
                        p["status"] = "CLOSED"
                        st.session_state["at_pnl"] += pnl
                        st.session_state["daily_pnl"] += pnl
                        st.session_state["at_log"].append(
                            f"[WS EXIT-SL] {sym} @ {sl:.2f} | PnL: {pnl:.2f}"
                        )
                        st.session_state["trade_history"].append({
                            "Timestamp": str(pd.Timestamp.now()),
                            "Symbol": sym,
                            "Side": "EXIT",
                            "Exit": sl,
                            "PnL": pnl,
                            "Reason": "SL",
                        })
                        update_equity_curve()

                    elif lp >= tp1:
                        pnl = (tp1 - entry) * qty
                        p["status"] = "CLOSED"
                        st.session_state["at_pnl"] += pnl
                        st.session_state["daily_pnl"] += pnl
                        st.session_state["at_log"].append(
                            f"[WS EXIT-TP] {sym} @ {tp1:.2f} | PnL: {pnl:.2f}"
                        )
                        st.session_state["trade_history"].append({
                            "Timestamp": str(pd.Timestamp.now()),
                            "Symbol": sym,
                            "Side": "EXIT",
                            "Exit": tp1,
                            "PnL": pnl,
                            "Reason": "TP",
                        })
                        update_equity_curve()

        # Optional trailing SL (requires df)
        # df_last = load_data(sym, "15m", years=1)
        # df_last = compute_indicators(df_last)
        # update_trailing_sl(sym, lp, df_last)

    except Exception:
        pass


# =========================
# UI LAYOUT
# =========================

st.set_page_config(page_title="Sniper Terminal v3", layout="wide")

st.title("Sniper Terminal v3 — Full Algo Mode")

tab_s, tab_b, tab_at, tab_algo, tab_g = st.tabs(
    ["Screener", "Backtest", "Auto Trade", "Algo Dashboard", "Guide"]
)

# -------------
# Screener (placeholder)
# -------------
with tab_s:
    st.markdown("### Screener")
    st.info("Plug your existing screener logic here.")


# -------------
# Backtest (placeholder)
# -------------
with tab_b:
    st.markdown("### Backtest")
    st.info("Plug your existing backtest engine here.")


# -------------
# AUTO TRADE TAB
# -------------
with tab_at:
    st.markdown("### Auto Trade (Paper + Shoonya Live)")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state["at_mode"] = st.selectbox(
            "Mode", ["paper", "shoonya"], index=0
        )
    with c2:
        st.session_state["risk_per_trade"] = st.number_input(
            "Risk per trade (INR)",
            value=float(st.session_state["risk_per_trade"]),
            step=500.0,
        )
    with c3:
        st.session_state["max_positions"] = st.number_input(
            "Max open positions",
            value=int(st.session_state["max_positions"]),
            step=1,
            min_value=1,
            max_value=50,
        )

    st.markdown("#### Shoonya Login")
    userid = st.text_input("User ID", key="sh_uid")
    pwd = st.text_input("Password", type="password", key="sh_pwd")
    pin = st.text_input("2FA PIN", type="password", key="sh_pin")
    vcode = st.text_input("Vendor Code", key="sh_vc")
    secret = st.text_input("API Secret", type="password", key="sh_sec")

    if st.button("Login to Shoonya"):
        try:
            res = st.session_state["shoonya"].login(
                userid=userid,
                password=pwd,
                twoFA=pin,
                vendor_code=vcode,
                api_secret=secret,
            )
            st.success("Shoonya Login Successful")
        except Exception as e:
            st.error(f"Login Failed: {e}")

    st.markdown("#### Build Token Map Automatically")

    if st.button("Download NSE Master & Build Token Map"):
        try:
            api = st.session_state["shoonya"]
            master = api.download_master("NSE")
            from shoonya_api import build_token_maps  # if you put it there
            token_map, symbol_map = build_token_maps(master)
            st.session_state["token_map"] = token_map
            st.session_state["symbol_map"] = symbol_map
            st.success(f"Token map built. {len(token_map)} instruments loaded.")
        except Exception as e:
            st.error(f"Failed to build token map: {e}")

    st.markdown("#### WebSocket & Algo Control")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Start WebSocket"):
            try:
                st.session_state["shoonya"].start_websocket(handle_shoonya_tick)
                st.success("WebSocket started.")
            except Exception as e:
                st.error(f"WebSocket error: {e}")
    with c2:
        if st.button("START FULL ALGO"):
            st.session_state["algo_running"] = True
            try:
                st.session_state["shoonya"].start_websocket(handle_shoonya_tick)
            except:
                pass
            st.success("Full Algo Mode Activated (Auto-Entry + Auto-Exit)")
    with c3:
        if st.button("STOP ALGO"):
            st.session_state["algo_running"] = False
            st.success("Algo Stopped.")

    st.markdown("### Advanced Algo Limits")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state["daily_loss_limit"] = st.number_input(
            "Daily loss limit (INR, negative)",
            value=float(st.session_state["daily_loss_limit"]),
            step=1000.0,
        )
    with c2:
        st.session_state["daily_profit_target"] = st.number_input(
            "Daily profit target (INR)",
            value=float(st.session_state["daily_profit_target"]),
            step=1000.0,
        )
    with c3:
        st.session_state["max_trades_per_symbol"] = st.number_input(
            "Max trades per symbol (per day)",
            value=int(st.session_state["max_trades_per_symbol"]),
            min_value=1,
            max_value=20,
            step=1,
        )

    st.caption(
        "Algo stops new entries if daily loss/profit limits hit, max positions reached, or per-symbol trade cap reached."
    )

    st.markdown("#### Open Positions")
    if st.session_state["at_pos"]:
        df_pos = pd.DataFrame(st.session_state["at_pos"]).T
        st.dataframe(df_pos, use_container_width=True, height=220)
    else:
        st.info("No open positions.")

    st.markdown("#### Auto Trade Log")
    if st.session_state["at_log"]:
        for line in reversed(st.session_state["at_log"][-50:]):
            st.markdown(f"- {line}")
    else:
        st.info("No activity yet.")

    st.metric("Cumulative PnL", f"INR {st.session_state['at_pnl']:.2f}")
    st.metric("Daily PnL", f"INR {st.session_state['daily_pnl']:.2f}")


# -------------
# ALGO DASHBOARD
# -------------
with tab_algo:
    st.markdown("### Algo Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algo Status", "RUNNING" if st.session_state["algo_running"] else "STOPPED")
    c2.metric("Open Positions", len([p for p in st.session_state["at_pos"].values() if p["status"] == "OPEN"]))
    c3.metric("Cumulative PnL", f"INR {st.session_state['at_pnl']:.2f}")
    c4.metric("Daily PnL", f"INR {st.session_state['daily_pnl']:.2f}")

    st.markdown("---")
    st.markdown("#### Live LTP (Tracked Symbols)")

    ltp_map = st.session_state["ltp"]
    if ltp_map:
        df_ltp = (
            pd.DataFrame(
                [{"Symbol": s, "LTP": round(p, 2)} for s, p in ltp_map.items()]
            )
            .sort_values("Symbol")
            .reset_index(drop=True)
        )
        st.dataframe(df_ltp, use_container_width=True, height=260)
    else:
        st.info("No live prices yet. Start WebSocket / Algo to populate LTP.")

    st.markdown("---")
    st.markdown("#### Open Positions (uPnL)")

    pos = st.session_state["at_pos"]
    if pos:
        rows = []
        for sym, p in pos.items():
            ltp = ltp_map.get(sym, p["entry"])
            direction = p["dir"]
            entry = p["entry"]
            sl = p["sl"]
            tp1 = p["tp1"]
            qty = p["qty"]
            status = p["status"]

            if direction == "BUY":
                upnl = (ltp - entry) * qty
            else:
                upnl = (entry - ltp) * qty

            rows.append(
                {
                    "Symbol": sym,
                    "Dir": direction,
                    "Status": status,
                    "Qty": qty,
                    "Entry": round(entry, 2),
                    "LTP": round(ltp, 2),
                    "SL": round(sl, 2),
                    "TP1": round(tp1, 2),
                    "uPnL": round(upnl, 2),
                }
            )

        df_pos = pd.DataFrame(rows)
        st.dataframe(df_pos, use_container_width=True, height=260)
    else:
        st.info("No open positions.")

    st.markdown("---")
    st.markdown("#### Recent Algo Activity")
    logs = st.session_state["at_log"]
    if logs:
        for line in reversed(logs[-50:]):
            st.markdown(f"- {line}")
    else:
        st.info("No algo activity yet.")

    st.markdown("---")
    st.markdown("### 🔥 Live Heatmap (Signals by Symbol × Timeframe)")

    timeframes = ["15m", "1h", "1d"]
    symbols = NIFTY50[:20]
    heatmap_data = []

    for sym in symbols:
        row = {"Symbol": sym}
        for tf in timeframes:
            try:
                df = load_data(sym, tf, years=1)
                if df.empty:
                    row[tf] = 0
                    continue
                df = compute_indicators(df)
                ltp = st.session_state["ltp"].get(sym, df["Close"].iloc[-1])
                score = live_signal_score(sym, ltp, df)
                row[tf] = score
            except Exception:
                row[tf] = 0
        heatmap_data.append(row)

    df_heat = pd.DataFrame(heatmap_data)
    fig_hm = px.imshow(
        df_heat[timeframes],
        labels=dict(x="Timeframe", y="Symbol", color="Signal Score"),
        x=timeframes,
        y=df_heat["Symbol"],
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )
    fig_hm.update_layout(
        height=500,
        template="plotly_dark",
        paper_bgcolor="#07090f",
        plot_bgcolor="#0d1117",
        font=dict(family="JetBrains Mono", color="#8b949e"),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚡ Live Breakout Map (AVWAP / EMA21 / EMA50 / PDH)")

    rows = []
    for sym in symbols:
        ltp = st.session_state["ltp"].get(sym)
        if not ltp:
            rows.append({"Symbol": sym, "AVWAP": 0, "EMA21": 0, "EMA50": 0, "PDH": 0})
            continue
        try:
            df = load_data(sym, "15m", years=1)
            if df.empty:
                rows.append({"Symbol": sym, "AVWAP": 0, "EMA21": 0, "EMA50": 0, "PDH": 0})
                continue
            df = compute_indicators(df)
            br = compute_breakout_status(sym, ltp, df)
            rows.append({
                "Symbol": sym,
                "AVWAP": br["AVWAP"],
                "EMA21": br["EMA21"],
                "EMA50": br["EMA50"],
                "PDH": br["PDH"],
            })
        except Exception:
            rows.append({"Symbol": sym, "AVWAP": 0, "EMA21": 0, "EMA50": 0, "PDH": 0})

    df_br = pd.DataFrame(rows)
    fig_br = px.imshow(
        df_br[["AVWAP", "EMA21", "EMA50", "PDH"]],
        labels=dict(x="Level", y="Symbol", color="Breakout"),
        x=["AVWAP", "EMA21", "EMA50", "PDH"],
        y=df_br["Symbol"],
        color_continuous_scale=[[0, "#111827"], [1, "#22c55e"]],
        zmin=0,
        zmax=1,
        aspect="auto",
    )
    fig_br.update_layout(
        height=500,
        template="plotly_dark",
        paper_bgcolor="#07090f",
        plot_bgcolor="#0d1117",
        font=dict(family="JetBrains Mono", color="#8b949e"),
    )
    st.plotly_chart(fig_br, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔥 Live Multi‑Breakout Strength Score (0–100)")

    rows = []
    for sym in symbols:
        ltp = st.session_state["ltp"].get(sym)
        if not ltp:
            rows.append({"Symbol": sym, "Strength": 0})
            continue
        try:
            df = load_data(sym, "15m", years=1)
            if df.empty:
                rows.append({"Symbol": sym, "Strength": 0})
                continue
            df = compute_indicators(df)
            strength = compute_multi_breakout_score(sym, ltp, df)
            rows.append({"Symbol": sym, "Strength": strength})
        except Exception:
            rows.append({"Symbol": sym, "Strength": 0})

    df_strength = pd.DataFrame(rows)
    fig_strength = px.imshow(
        df_strength[["Strength"]],
        labels=dict(x="Metric", y="Symbol", color="Strength"),
        x=["Strength"],
        y=df_strength["Symbol"],
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=100,
        aspect="auto",
    )
    fig_strength.update_layout(
        height=500,
        template="plotly_dark",
        paper_bgcolor="#07090f",
        plot_bgcolor="#0d1117",
        font=dict(family="JetBrains Mono", color="#8b949e"),
    )
    st.plotly_chart(fig_strength, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Sector Strength Model")

    sector_strength = compute_sector_strength()
    df_sector = pd.DataFrame(
        [{"Sector": s, "Strength": round(v, 2)} for s, v in sector_strength.items()]
    )
    st.dataframe(df_sector, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🌐 Market Breadth")

    mb = compute_market_breadth()
    c1, c2, c3 = st.columns(3)
    c1.metric("Advancers", mb["adv"])
    c2.metric("Decliners", mb["dec"])
    c3.metric("Breadth Score", f"{mb['breadth']:.1f}")

    st.markdown("---")
    st.markdown("### 📈 Equity Curve & Drawdown")

    dd = compute_drawdown()
    c1, c2 = st.columns(2)
    c1.metric("Max Equity", f"INR {st.session_state['max_equity']:.2f}")
    c2.metric("Drawdown", f"{dd:.2f}%")

    if st.session_state["equity_curve"]:
        df_eq = pd.DataFrame(st.session_state["equity_curve"])
        df_eq.set_index("ts", inplace=True)
        st.line_chart(df_eq["equity"])
    else:
        st.info("No trades yet to build equity curve.")

    st.markdown("---")
    st.markdown("### ⚠ Volatility Auto-Pause")

    c1, c2 = st.columns(2)
    c1.metric("Volatility Regime", st.session_state.get("vol_regime", "NORMAL"))
    if c2.button("Resume Algo (Clear Auto-Pause)"):
        st.session_state["auto_pause"] = False
        st.success("Auto-pause cleared. Algo can enter again.")

    st.markdown("---")
    st.markdown("### ML Filter")

    if st.button("Train ML Filter (Last 6 Months)"):
        X_train = []
        y_train = []
        for sym in NIFTY50:
            try:
                df = load_data(sym, "1d", years=1)
                if df.empty:
                    continue
                df = compute_indicators(df)
                bull, _ = compute_divergences(df, bars=5)
                has_div = 1 if bull else 0
                strength = compute_multi_breakout_score(sym, df["Close"].iloc[-1], df)
                X_train.append(ml_features(df, strength, has_div)[0])
                y_train.append(1 if df["Close"].iloc[-1] > df["EMA200"].iloc[-1] else 0)
            except Exception:
                continue
        if X_train:
            st.session_state["ml_model"].fit(X_train, y_train)
            st.session_state["ml_trained"] = True
            st.success("ML Filter trained successfully")
        else:
            st.warning("No data available to train ML filter.")

    st.markdown("---")
    st.markdown("### Trade Log Export")

    if st.button("Export Trade Log (CSV)"):
        df = pd.DataFrame(st.session_state["trade_history"])
        if df.empty:
            st.warning("No trades yet.")
        else:
            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                file_name="sniper_trades.csv",
                mime="text/csv",
            )


# -------------
# GUIDE TAB
# -------------
with tab_g:
    st.markdown("### Guide")
    st.write(
        "- Configure risk, limits, and mode in Auto Trade tab.\n"
        "- Start WebSocket and Full Algo.\n"
        "- Monitor everything in Algo Dashboard.\n"
        "- Use trade log export for journaling and analysis."
    )


# =========================
# BACKGROUND LOGIC PER REFRESH
# =========================

compute_vol_regime()

if st.session_state.get("algo_running", False):
    auto_entry_engine()
