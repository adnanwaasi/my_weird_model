"""Train an LSTM to predict a stock's next-day return and evaluate it honestly.

Usage:
    python train.py --ticker AAPL
    python train.py --ticker RELIANCE.NS --start 2015-01-01 --epochs 50
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import FEATURES, TARGET, Scaler, chronological_split, download_prices, make_features, make_windows
from model import LSTMRegressor
from predict import forecast_from_window


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--window", type=int, default=30, help="days of history per prediction")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--patience", type=int, default=8, help="early-stopping patience in epochs")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def predict(model, x, device):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x).to(device)).cpu().numpy()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.ticker} from {args.start}...")
    feats = make_features(download_prices(args.ticker, args.start, args.end))
    latest = feats[FEATURES].dropna().iloc[-args.window :]  # for tomorrow's forecast
    feats = feats.dropna()
    train_df, val_df, test_df = chronological_split(feats)
    print(f"{len(feats)} days -> train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")

    x_scaler = Scaler.fit(train_df[FEATURES].values)
    y_scaler = Scaler.fit(train_df[[TARGET]].values)

    def windows(df):
        x = x_scaler.transform(df[FEATURES].values)
        y = y_scaler.transform(df[[TARGET]].values).ravel()
        return make_windows(x, y, args.window)

    x_train, y_train = windows(train_df)
    x_val, y_val = windows(val_df)
    x_test, y_test = windows(test_df)
    if min(len(x_train), len(x_val), len(x_test)) == 0:
        raise SystemExit("Not enough data for this window size; use an earlier --start or smaller --window.")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,  # shuffling whole windows is fine; each window is self-contained
    )
    model = LSTMRegressor(len(FEATURES), hidden_size=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(xb)
        val_loss = float(np.mean((predict(model, x_val, device) - y_val) ** 2))
        print(f"epoch {epoch:3d}  train {total / len(x_train):.4f}  val {val_loss:.4f}")
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no val improvement for {args.patience} epochs)")
                break
    model.load_state_dict(best_state)

    # --- Evaluate on the untouched test set, in real return units ---
    pred = y_scaler.inverse(predict(model, x_test, device)[:, None]).ravel()
    actual = y_scaler.inverse(y_test[:, None]).ravel()
    rmse_model = np.sqrt(np.mean((pred - actual) ** 2))
    rmse_zero = np.sqrt(np.mean(actual**2))  # baseline: "tomorrow's return is 0"
    dir_acc = np.mean(np.sign(pred) == np.sign(actual))
    always_up = np.mean(actual > 0)  # baseline: "the stock always goes up"
    strategy = np.cumsum(np.where(pred > 0, actual, 0.0))  # long when predicting up, else cash
    buy_hold = np.cumsum(actual)

    print("\n=== Test set results ===")
    print(f"RMSE model            {rmse_model:.5f}")
    print(f"RMSE predict-zero     {rmse_zero:.5f}   (model must beat this to be useful)")
    print(f"Direction accuracy    {dir_acc:.1%}")
    print(f"Always-up baseline    {always_up:.1%}")
    print(f"Strategy log return   {strategy[-1]:+.3f}  vs buy & hold {buy_hold[-1]:+.3f}  (no costs/slippage)")

    # --- Tomorrow's forecast from the most recent window ---
    if len(latest) == args.window:
        nxt = forecast_from_window(model, x_scaler, y_scaler, latest.values)
        print(f"\nForecast for the trading day after {latest.index[-1].date()}: "
              f"{np.expm1(nxt):+.2%} ({'UP' if nxt > 0 else 'DOWN'})")

    # --- Save model + plots ---
    ckpt = out_dir / f"{args.ticker}_lstm.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_mean": x_scaler.mean, "x_std": x_scaler.std,
            "y_mean": y_scaler.mean, "y_std": y_scaler.std,
            "features": FEATURES, "args": vars(args),
        },
        ckpt,
    )

    dates = test_df.index[args.window - 1 :]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(dates, actual, label="actual", alpha=0.6, linewidth=0.8)
    ax1.plot(dates, pred, label="predicted", linewidth=1.2)
    ax1.set_ylabel("next-day log return")
    ax1.set_title(f"{args.ticker} LSTM: test set")
    ax1.legend()
    ax2.plot(dates, buy_hold, label="buy & hold")
    ax2.plot(dates, strategy, label="LSTM long/cash")
    ax2.set_ylabel("cumulative log return")
    ax2.legend()
    fig.tight_layout()
    plot = out_dir / f"{args.ticker}_test.png"
    fig.savefig(plot, dpi=120)
    print(f"\nSaved {ckpt} and {plot}")


if __name__ == "__main__":
    main()
