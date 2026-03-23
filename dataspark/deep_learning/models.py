"""
PyTorch Models
==============
Tabular MLP and LSTM-based time-series forecaster.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """Multi-layer perceptron for tabular classification/regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = (256, 128, 64),
        output_dim: int = 1,
        dropout: float = 0.3,
        task: str = "classification",
    ) -> None:
        super().__init__()
        self.task = task
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        if self.task == "classification" and logits.shape[-1] > 1:
            return logits  # raw logits — use CrossEntropyLoss
        return logits.squeeze(-1)


class LSTMForecaster(nn.Module):
    """LSTM for univariate/multivariate time-series forecasting."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # last time step
