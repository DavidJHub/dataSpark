"""
Neural Network Trainer
======================
Generic PyTorch training loop with early stopping, LR scheduling,
and experiment tracking.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger


class NeuralNetTrainer:
    """Train PyTorch models with a configurable loop."""

    def __init__(
        self,
        model: nn.Module,
        task: Literal["classification", "regression"] = "classification",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.task = task
        self.patience = patience

        self.criterion = (
            nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        )
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=patience // 2, factor=0.5
        )
        self.history: dict[str, list] = {"train_loss": [], "val_loss": []}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 100,
        batch_size: int = 256,
    ) -> dict[str, list]:
        """Train the model and return training history."""
        train_loader = self._make_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, batch_size) if X_val is not None else None

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self._eval_epoch(val_loader)
                self.history["val_loss"].append(val_loss)
                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch {}", epoch)
                    break

                if epoch % 10 == 0:
                    logger.info(
                        "Epoch {}/{} — train_loss: {:.4f}, val_loss: {:.4f}",
                        epoch, epochs, train_loss, val_loss,
                    )
            else:
                if epoch % 10 == 0:
                    logger.info("Epoch {}/{} — train_loss: {:.4f}", epoch, epochs, train_loss)

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        self.model.eval()
        tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        if self.task == "classification":
            if output.dim() > 1 and output.shape[-1] > 1:
                return output.argmax(dim=-1).cpu().numpy()
            return (output > 0).int().cpu().numpy()
        return output.cpu().numpy()

    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(X_batch)
            loss = self.criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    def _make_loader(
        self, X: np.ndarray | None, y: np.ndarray | None, batch_size: int, shuffle: bool = False
    ) -> DataLoader | None:
        if X is None:
            return None
        X_t = torch.FloatTensor(X)
        if self.task == "classification":
            y_t = torch.LongTensor(y)
        else:
            y_t = torch.FloatTensor(y)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)
