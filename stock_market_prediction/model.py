import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """Stacked LSTM that maps a window of daily features to one next-day return."""

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features) -> (batch,)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)
