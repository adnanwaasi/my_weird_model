"""Data loading, feature engineering and windowing for the LSTM stock predictor.

The model predicts the *next-day log return*, not the raw price. Predicting the
raw price makes any model look great (tomorrow's price ~= today's price), which
hides whether it learned anything. Returns force an honest comparison against
naive baselines.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = ["log_return", "hl_range", "volume_change", "ma10_gap", "ma30_gap"]
TARGET = "next_log_return"


def download_prices(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker!r}")
    # Recent yfinance versions return (field, ticker) MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build model features from OHLCV data.

    Every feature on row t uses only information available at the close of day t.
    The target on row t is the log return from day t to day t+1.
    """
    close = prices["Close"]
    out = pd.DataFrame(index=prices.index)
    out["log_return"] = np.log(close).diff()
    out["hl_range"] = (prices["High"] - prices["Low"]) / close
    out["volume_change"] = np.log(prices["Volume"].replace(0, np.nan)).diff()
    out["ma10_gap"] = close / close.rolling(10).mean() - 1
    out["ma30_gap"] = close / close.rolling(30).mean() - 1
    out[TARGET] = out["log_return"].shift(-1)
    return out.replace([np.inf, -np.inf], np.nan)


def chronological_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15):
    """Split by time (never shuffle time series): oldest -> train, newest -> test."""
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    return df.iloc[:n_train], df.iloc[n_train : n_train + n_val], df.iloc[n_train + n_val :]


@dataclass
class Scaler:
    """Standard scaler fitted on training data only, to avoid look-ahead leakage."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        std = x.std(axis=0)
        return cls(mean=x.mean(axis=0), std=np.where(std == 0, 1.0, std))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def make_windows(x: np.ndarray, y: np.ndarray, window: int):
    """Turn a (T, F) feature matrix into (N, window, F) sequences.

    Sequence i covers rows [i, i + window) and is labelled with y at its last row,
    i.e. the return of the day *after* the window ends.
    """
    if len(x) < window:
        return np.empty((0, window, x.shape[1]), dtype=np.float32), np.empty(0, dtype=np.float32)
    idx = np.arange(window)[None, :] + np.arange(len(x) - window + 1)[:, None]
    return x[idx].astype(np.float32), y[window - 1 :].astype(np.float32)
