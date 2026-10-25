import numpy as np
import pandas as pd
import torch

from data import FEATURES, TARGET, Scaler, chronological_split, make_features, make_windows
from model import LSTMRegressor


def fake_prices(n=200, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1_000, 10_000, n).astype(float),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def test_target_is_next_day_return():
    feats = make_features(fake_prices())
    np.testing.assert_allclose(feats[TARGET].iloc[:-1].values, feats["log_return"].iloc[1:].values)
    assert np.isnan(feats[TARGET].iloc[-1])


def test_features_do_not_use_future_prices():
    prices = fake_prices()
    full = make_features(prices)
    truncated = make_features(prices.iloc[:100])
    # Changing the future must not change features computed for the past.
    pd.testing.assert_frame_equal(full[FEATURES].iloc[:100], truncated[FEATURES])


def test_chronological_split_keeps_order_and_covers_everything():
    df = make_features(fake_prices()).dropna()
    train, val, test = chronological_split(df)
    assert train.index.max() < val.index.min() <= val.index.max() < test.index.min()
    assert len(train) + len(val) + len(test) == len(df)


def test_scaler_roundtrip_and_constant_column():
    x = np.array([[1.0, 5.0], [3.0, 5.0]])
    s = Scaler.fit(x)
    np.testing.assert_allclose(s.inverse(s.transform(x)), x)
    assert np.all(np.isfinite(s.transform(x)))


def test_windows_are_labelled_with_last_row_target():
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float) * 10
    xw, yw = make_windows(x, y, window=3)
    assert xw.shape == (8, 3, 2)
    np.testing.assert_array_equal(xw[0], x[0:3])
    assert yw[0] == y[2]
    np.testing.assert_array_equal(xw[-1], x[7:10])
    assert yw[-1] == y[9]


def test_windows_shorter_than_window_is_empty():
    xw, yw = make_windows(np.zeros((2, 3)), np.zeros(2), window=5)
    assert xw.shape == (0, 5, 3) and yw.shape == (0,)


def test_model_output_shape():
    model = LSTMRegressor(n_features=len(FEATURES))
    assert model(torch.zeros(4, 30, len(FEATURES))).shape == (4,)
