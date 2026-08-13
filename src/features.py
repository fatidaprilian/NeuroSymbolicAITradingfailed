"""
Feature Engineering Module for Neuro-Symbolic Crypto Trading Framework
Standardizes technical indicator calculations and dataset splits across BTC, ETH, XRP.
"""

import pandas as pd
import numpy as np
import os


FEATURE_COLUMNS = [
    'close', 'volume', 'SMA_7', 'SMA_30', 'EMA_12', 'EMA_26',
    'MACD', 'MACD_signal', 'RSI', 'BB_upper', 'BB_lower', 'ATR'
]


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes standard technical indicators on price DataFrame.
    DataFrame must contain: ['close', 'high', 'low', 'volume']
    """
    df = df.copy()
    df = df.ffill()

    # Moving Averages
    df['SMA_7'] = df['close'].rolling(window=7).mean()
    df['SMA_30'] = df['close'].rolling(window=30).mean()
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI (14-period)
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)

    # Bollinger Bands (20-period)
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (2 * df['BB_std'])
    df['BB_lower'] = df['BB_middle'] - (2 * df['BB_std'])

    # Average True Range (ATR 14-period) & Normalized ATR (NATR %)
    if 'high' in df.columns and 'low' in df.columns:
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
    else:
        df['ATR'] = df['close'].rolling(window=14).std()

    df['NATR'] = (df['ATR'] / df['close']) * 100.0

    # Target: Next period close price
    df['target'] = df['close'].shift(-1)

    df.dropna(inplace=True)
    return df


def load_and_preprocess_data(symbol: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Loads raw CSV data for a symbol (btc, eth, xrp) and computes features.
    Checks data/, ., and parent paths.
    """
    symbol_lower = symbol.lower()
    possible_paths = [
        os.path.join(data_dir, f"{symbol_lower}_1h_data.csv"),
        f"{symbol_lower}_1h_data.csv",
        os.path.join("..", data_dir, f"{symbol_lower}_1h_data.csv")
    ]

    filename = None
    for p in possible_paths:
        if os.path.exists(p):
            filename = p
            break

    if filename is None:
        raise FileNotFoundError(f"Data file for [{symbol}] not found in paths: {possible_paths}")

    df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
    df = compute_technical_indicators(df)
    return df


def train_val_test_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """
    Chronological data split for time-series forecasting without data leakage.
    Returns: df_train, df_val, df_test
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    return df_train, df_val, df_test
