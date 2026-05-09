import numpy as np
import pandas as pd
from .indicators import ema, rsi, atr, avwap

def detect_strict_swing_lows(df):
    lows = df['Low'].values
    swing_low = np.zeros(len(df), dtype=bool)

    for i in range(2, len(df) - 2):
        if (
            lows[i] < lows[i - 1] and
            lows[i] < lows[i - 2] and
            lows[i] < lows[i + 1] and
            lows[i] < lows[i + 2]
        ):
            swing_low[i] = True

    return swing_low

def detect_rsi_bullish_divergence(df, swing_low_mask):
    divergence_points = []
    lows = df['Low'].values
    rsi_vals = df['RSI'].values

    swing_indices = np.where(swing_low_mask)[0]

    for i in range(1, len(swing_indices)):
        i1 = swing_indices[i - 1]
        i2 = swing_indices[i]

        if lows[i2] < lows[i1] and rsi_vals[i2] > rsi_vals[i1]:
            divergence_points.append((i1, i2))

    return divergence_points

def generate_divergence_markers(df, divergence_pairs):
    df['Div_Arrow'] = np.nan
    df['Div_Line_Price'] = np.nan
    df['Div_Line_RSI'] = np.nan

    for (i1, i2) in divergence_pairs:
        df.iloc[i2, df.columns.get_loc('Div_Arrow')] = df['Low'].iloc[i2] * 0.995

        df.iloc[i1:i2+1, df.columns.get_loc('Div_Line_Price')] = np.linspace(
            df['Low'].iloc[i1],
            df['Low'].iloc[i2],
            i2 - i1 + 1
        )

        df.iloc[i1:i2+1, df.columns.get_loc('Div_Line_RSI')] = np.linspace(
            df['RSI'].iloc[i1],
            df['RSI'].iloc[i2],
            i2 - i1 + 1
        )

    return df

def apply_divergence_engine(df):
    df['EMA200'] = ema(df['Close'], 200)
    df['RSI'] = rsi(df['Close'])
    df['ATR'] = atr(df)
    df['AVWAP'] = avwap(df)

    swing_mask = detect_strict_swing_lows(df)
    df['SwingLow'] = swing_mask

    div_pairs = detect_rsi_bullish_divergence(df, swing_mask)

    df = generate_divergence_markers(df, div_pairs)

    return df, div_pairs
