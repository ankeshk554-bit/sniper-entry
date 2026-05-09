import numpy as np
import pandas as pd

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    gain_ema = gain.ewm(span=length, adjust=False).mean()
    loss_ema = loss.ewm(span=length, adjust=False).mean()

    rs = gain_ema / loss_ema
    return 100 - (100 / (1 + rs))

def atr(df, length=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()

def avwap(df):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cumulative_tp_vol = (typical_price * df['Volume']).cumsum()
    cumulative_vol = df['Volume'].cumsum()
    return cumulative_tp_vol / cumulative_vol
