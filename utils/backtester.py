import numpy as np
import pandas as pd

def run_backtest(df, divergence_pairs, risk_per_trade=2000):
    # Clean and stabilize dataframe
    df = df.copy()
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    df = df.reset_index().rename(columns={"index": "Timestamp"})

    trades = []
    equity = 0

    # We IGNORE divergence_pairs now and use Div_Arrow as entry signal
    for i in range(len(df)):
        # Entry only where divergence arrow exists
        if pd.isna(df.at[i, "Div_Arrow"]):
            continue

        close_i = float(df.at[i, "Close"])
        ema_i = float(df.at[i, "EMA200"])
        avwap_i = float(df.at[i, "AVWAP"])
        atr_i = float(df.at[i, "ATR"])

        if (
            np.isnan(close_i)
            or np.isnan(ema_i)
            or np.isnan(avwap_i)
            or np.isnan(atr_i)
        ):
            continue

        # Trend filter
        if close_i < ema_i:
            continue

        # AVWAP filter
        if close_i < avwap_i:
            continue

        entry_price = close_i

        if atr_i <= 0:
            continue

        sl_distance = 1.5 * atr_i
        qty = max(int(risk_per_trade / sl_distance), 1)

        sl_price = entry_price - sl_distance
        tp_price = entry_price + (2 * atr_i)

        exit_price = None
        exit_index = None

        for j in range(i + 1, len(df)):
            low_j = float(df.at[j, "Low"])
            high_j = float(df.at[j, "High"])

            if np.isnan(low_j) or np.isnan(high_j):
                continue

            if low_j <= sl_price:
                exit_price = sl_price
                exit_index = j
                break

            if high_j >= tp_price:
                exit_price = tp_price
                exit_index = j
                break

        if exit_price is None:
            exit_price = float(df.at[len(df) - 1, "Close"])
            exit_index = len(df) - 1

        pnl = (exit_price - entry_price) * qty
        equity += pnl

        trades.append({
            "Entry": df.at[i, "Timestamp"],
            "Exit": df.at[exit_index, "Timestamp"],
            "EntryPrice": entry_price,
            "ExitPrice": exit_price,
            "Qty": qty,
            "PnL": pnl,
            "Equity": equity
        })

    return trades, equity


def trades_to_df(trades):
    if len(trades) == 0:
        return pd.DataFrame(columns=[
            "Entry", "Exit", "EntryPrice", "ExitPrice", "Qty", "PnL", "Equity"
        ])
    return pd.DataFrame(trades)


def build_equity_curve(trades_df):
    if len(trades_df) == 0:
        return pd.Series(dtype=float)
    return trades_df["Equity"]
