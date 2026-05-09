import numpy as np
import pandas as pd

def run_backtest(df, divergence_pairs, risk_per_trade=2000):
    # ⭐ FIX 1: remove duplicate timestamps
    df = df[~df.index.duplicated(keep='first')].copy()
    df = df.sort_index()

    # ⭐ FIX 2: reset index to guarantee integer indexing works
    df = df.reset_index(drop=False)

    trades = []
    equity = 0

    # Convert divergence pairs into entry indices
    divergence_entries = {i2 for (_, i2) in divergence_pairs}

    for i in range(len(df)):
        if i not in divergence_entries:
            continue

        close_i = df['Close'].iloc[i]
        ema_i = df['EMA200'].iloc[i]
        avwap_i = df['AVWAP'].iloc[i]
        atr_i = df['ATR'].iloc[i]

        # ⭐ FIX 3: skip invalid rows
        if (
            pd.isna(close_i) or pd.isna(ema_i) or
            pd.isna(avwap_i) or pd.isna(atr_i)
        ):
            continue

        # Trend filter
        if close_i < ema_i:
            continue

        # AVWAP filter
        if close_i < avwap_i:
            continue

        entry_price = close_i

        if atr_i == 0:
            continue

        sl_distance = 1.5 * atr_i
        qty = max(int(risk_per_trade / sl_distance), 1)

        sl_price = entry_price - sl_distance
        tp_price = entry_price + (2 * atr_i)

        exit_price = None
        exit_index = None

        # ⭐ FIX 4: safe forward simulation
        for j in range(i + 1, len(df)):
            low_j = df['Low'].iloc[j]
            high_j = df['High'].iloc[j]

            if pd.isna(low_j) or pd.isna(high_j):
                continue

            if low_j <= sl_price:
                exit_price = sl_price
                exit_index = j
                break

            if high_j >= tp_price:
                exit_price = tp_price
                exit_index = j
                break

        # If no exit found
        if exit_price is None:
            exit_price = df['Close'].iloc[-1]
            exit_index = len(df) - 1

        pnl = (exit_price - entry_price) * qty
        equity += pnl

        trades.append({
            "Entry": df['index'].iloc[i],   # original timestamp
            "Exit": df['index'].iloc[exit_index],
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
