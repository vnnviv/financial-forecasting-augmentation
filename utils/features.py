"""
Feature engineering for financial time series.

8 features derived from Close price:
  RSI_14, SMA_5/10/20, Vol_20, Mom_5, BB_pos

Key rule: all features recomputed from scratch on each series
(real or synthetic). Never inherit features from real data
onto synthetic prices — that's a leakage channel.

Vivian Chan | 2026
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


FEATURE_COLS = [
    'Close', 'RSI_14', 'SMA_5', 'SMA_10', 'SMA_20',
    'Vol_20', 'Mom_5', 'BB_pos'
]


def compute_rsi(prices, period=14):
    """14-day RSI. squeeze() call keeps it 1D regardless of df shape."""
    prices   = prices.squeeze()
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))


def compute_features(df):
    """
    Takes a DataFrame with a 'Close' column.
    Returns the same DataFrame with 7 new feature columns.

    Drops the first ~20 rows because rolling windows need warmup.
    """
    df    = df.copy()
    close = df['Close'].squeeze()

    df['RSI_14'] = compute_rsi(close, 14)
    df['SMA_5']  = close.rolling(5).mean()
    df['SMA_10'] = close.rolling(10).mean()
    df['SMA_20'] = close.rolling(20).mean()

    returns      = close.pct_change()
    df['Vol_20'] = returns.rolling(20).std() * np.sqrt(252)  # annualized
    df['Mom_5']  = close.pct_change(5)

    bb_mean      = close.rolling(20).mean()
    bb_std       = close.rolling(20).std()
    df['BB_pos'] = (close - bb_mean) / (2 * bb_std + 1e-8)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def temporal_split(df, train_ratio=0.65, val_ratio=0.15):
    """
    Strict chronological split — no shuffling.
    train=65% | val=15% | test=20%

    This is the fix that Part 1 was missing. Random splitting
    on autocorrelated data = data leakage.
    """
    n  = len(df)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    return (
        df.iloc[:t1].copy(),
        df.iloc[t1:t2].copy(),
        df.iloc[t2:].copy()
    )


def fit_scaler(train_df):
    """
    Fit MinMaxScaler on training data only.
    Fitting on the full series leaks future price range into training.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_df[FEATURE_COLS].values)
    return scaler


def apply_scaler(scaler, df):
    return scaler.transform(df[FEATURE_COLS].values)


def create_sequences(data, seq_length=20):
    """20-day rolling window sequences for LSTM input."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Close only
    return np.array(X), np.array(y)
