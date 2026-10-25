"""Load a trained checkpoint and forecast the next trading day's return.

Usage:
    python predict.py --ticker AAPL            # uses artifacts/AAPL_lstm.pt
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

from data import FEATURES, Scaler, download_prices, make_features
from model import LSTMRegressor

ARTIFACTS = Path(__file__).parent / "artifacts"


def forecast_from_window(model, x_scaler: Scaler, y_scaler: Scaler, window_features: np.ndarray) -> float:
    """Predict the next-day log return from the last `window` rows of raw features."""
    x = x_scaler.transform(window_features)[None].astype(np.float32)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
    return float(y_scaler.inverse(pred[:, None]).item())


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    model = LSTMRegressor(len(ckpt["features"]), hidden_size=args["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    x_scaler = Scaler(ckpt["x_mean"], ckpt["x_std"])
    y_scaler = Scaler(ckpt["y_mean"], ckpt["y_std"])
    return model, x_scaler, y_scaler, args


def forecast(ticker: str, checkpoint: Path | None = None) -> dict:
    """Forecast the next trading day for `ticker` using its trained checkpoint."""
    path = checkpoint or ARTIFACTS / f"{ticker}_lstm.pt"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}. Train one first: python train.py --ticker {ticker}")
    model, x_scaler, y_scaler, args = load_checkpoint(path)
    window = args["window"]
    # ~3x the window in calendar days covers weekends, holidays and the 30-day moving average warm-up.
    start = (date.today() - timedelta(days=window * 3 + 60)).isoformat()
    latest = make_features(download_prices(ticker, start))[FEATURES].dropna().iloc[-window:]
    if len(latest) < window:
        raise ValueError(f"Only {len(latest)} recent rows for {ticker}; need {window}")
    log_ret = forecast_from_window(model, x_scaler, y_scaler, latest.values)
    return {
        "ticker": ticker,
        "as_of": str(latest.index[-1].date()),
        "predicted_next_day_return": float(np.expm1(log_ret)),
        "direction": "UP" if log_ret > 0 else "DOWN",
        "note": "Single-stock LSTM on price/volume features only; treat as a weak signal, not advice.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--checkpoint", type=Path, default=None)
    a = p.parse_args()
    r = forecast(a.ticker, a.checkpoint)
    print(f"{r['ticker']} after {r['as_of']}: {r['predicted_next_day_return']:+.2%} ({r['direction']})")
