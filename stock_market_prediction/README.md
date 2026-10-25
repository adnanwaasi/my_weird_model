# LSTM stock predictor

Predicts a stock's **next-day log return** from 30 days of price and volume features. It is evaluated
honestly against naive baselines on a chronologically held-out test set.

```bash
pip install -r requirements.txt
python train.py --ticker AAPL            # trains, prints test metrics, saves artifacts/AAPL_lstm.pt + plot
python predict.py --ticker AAPL          # next-day forecast from the saved checkpoint
pytest                                   # leakage / windowing / shape tests (no network)
```

How to read the results:
- **RMSE vs predict-zero:** if the model doesn't beat "tomorrow's return is 0", it learned nothing useful.
  This is the normal outcome for daily returns.
- **Direction accuracy vs always-up:** only meaningful if it beats the always-up baseline.
- **Strategy return** ignores trading costs and slippage.

Why returns and not prices: predicting raw prices gives a beautiful chart for any model, because
tomorrow's price ≈ today's price. That hides whether the model learned anything.

The research agent (`../market_research_agent`) calls `predict.forecast()` as its `lstm_forecast` tool.
