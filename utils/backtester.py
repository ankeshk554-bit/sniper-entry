import numpy as np
import pandas as pd

def run_backtest(df, divergence_pairs, risk_per_trade=2000):
    trades = []
    equity = 0
    df = df.copy()

    divergence_entries = {i2 for (_, i2) in divergence_pairs}

    for i in range(len(df)):
        if i not in divergence_entries:
            continue

        if df['Close'].iloc[i] < df['EMA200'].iloc[i]:
            continue

        if df['Close'].iloc[i] < df['AVWAP'].iloc[i]:
            continue

        entry_price = df['Close'].iloc[i]
        atr_val = df['ATR'].iloc[i]

        if np.isnan(atr_val) or atr_val == 0:
            continue

        sl_distance = 1.5 * atr_val
        qty = max(int(risk_per_trade / sl_distance), 1)

        sl_price = entry_price - sl_distance
        tp_price = entry_price + (2 * atr_val)

        exit_price = None
        exit_index = None

        for j in range(i + 1, len(df)):
            low_j = df['Low'].iloc[j]
            high_j = df['High'].iloc[j]

            if low_j <= sl_price:
                exit_price = sl_price
                exit_index = j
                break

            if high_j >= tp_price:
                exit_price = tp_price
                exit_index = j
                break

        if exit_price is None:
            exit_price = df['Close'].iloc[-1]
            exit_index = len(df) - 1

        pnl = (exit_price - entry_price) * qty
        equity += pnl

        trades.append({
            "Entry": df.index[i],
            "Exit": df.index[exit_index],
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
