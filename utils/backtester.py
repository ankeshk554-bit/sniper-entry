import numpy as np
import pandas as pd

def run_backtest(df, divergence_pairs, risk_per_trade=2000):
    # FULL HARD RESET OF INDEX (fixes all Series comparison errors)
    df = df.copy()
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    df = df.reset_index().rename(columns={"index": "Timestamp"})

    trades = []
    equity = 0

    # Convert divergence pairs into entry indices
    divergence_entries = {i2 for (_, i2) in divergence_pairs}

    for i in range(len(df)):
        if i not in divergence_entries:
            continue

        # Extract scalars safely
        close_i = float(df.at[i, "Close"])
        ema_i = float(df.at[i, "EMA200"])
        avwap_i = float(df.at[i, "AVWAP"])
        atr_i = float(df.at[i, "ATR"])

        # Skip invalid rows
        if np.isnan(close_i) or np.isnan(ema_i) or np.isnan(avwap_i) or np.isnan(atr_i):
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

        # Forward simulation
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

        # No exit found → exit at last close
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
