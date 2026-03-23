"""Tests for dataspark.deep_learning module."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
from dataspark.deep_learning import TabularMLP, LSTMForecaster, NeuralNetTrainer


class TestTabularMLP:
    def test_forward_classification(self):
        import torch
        model = TabularMLP(input_dim=10, hidden_dims=[32, 16], output_dim=3, task="classification")
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 3)

    def test_forward_regression(self):
        import torch
        model = TabularMLP(input_dim=5, hidden_dims=[16], output_dim=1, task="regression")
        x = torch.randn(4, 5)
        out = model(x)
        assert out.shape == (4,)


class TestLSTMForecaster:
    def test_forward(self):
        import torch
        model = LSTMForecaster(input_dim=1, hidden_dim=16, num_layers=1, output_dim=1)
        x = torch.randn(4, 10, 1)  # batch=4, seq_len=10, features=1
        out = model(x)
        assert out.shape == (4, 1)


class TestNeuralNetTrainer:
    def test_fit_and_predict(self):
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int64)
        model = TabularMLP(input_dim=5, hidden_dims=[16, 8], output_dim=2, task="classification")
        trainer = NeuralNetTrainer(model, task="classification", lr=1e-2, patience=5)
        history = trainer.fit(X, y, X_val=X, y_val=y, epochs=20, batch_size=32)
        assert len(history["train_loss"]) > 0
        preds = trainer.predict(X)
        assert len(preds) == 100

    def test_regression(self):
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        y = (X[:, 0] * 2 + X[:, 1]).astype(np.float32)
        model = TabularMLP(input_dim=3, hidden_dims=[16], output_dim=1, task="regression")
        trainer = NeuralNetTrainer(model, task="regression", lr=1e-2, patience=5)
        trainer.fit(X, y, epochs=20, batch_size=32)
        preds = trainer.predict(X)
        assert len(preds) == 100
