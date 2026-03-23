"""
Forecasting Module
==================
ARIMA, Exponential Smoothing, and rolling-mean baselines.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger


class Forecaster:
    """Time-series forecasting with multiple methods."""

    def __init__(
        self,
        method: Literal["arima", "exp_smoothing", "rolling_mean"] = "arima",
    ) -> None:
        self.method = method
        self._model = None

    def fit(self, series: pd.Series, **kwargs) -> "Forecaster":
        """Fit the chosen model."""
        series = series.dropna()
        if self.method == "arima":
            self._fit_arima(series, **kwargs)
        elif self.method == "exp_smoothing":
            self._fit_exp_smoothing(series, **kwargs)
        elif self.method == "rolling_mean":
            self._window = kwargs.get("window", 12)
        logger.info("Forecaster fitted (method={})", self.method)
        return self

    def predict(self, steps: int = 12) -> pd.Series:
        """Forecast future values."""
        if self.method == "rolling_mean":
            # naive: repeat last rolling mean
            return pd.Series([self._last_rolling_mean] * steps)
        forecast = self._model.forecast(steps)
        return forecast

    def evaluate(self, train: pd.Series, test: pd.Series) -> dict:
        """Fit on train, predict test length, return error metrics."""
        self.fit(train)
        preds = self.predict(steps=len(test))
        preds.index = test.index
        mae = np.mean(np.abs(test - preds))
        rmse = np.sqrt(np.mean((test - preds) ** 2))
        mape = np.mean(np.abs((test - preds) / test.replace(0, np.nan))) * 100
        return {"mae": mae, "rmse": rmse, "mape": mape, "n_test": len(test)}

    def _fit_arima(self, series: pd.Series, order: tuple = (1, 1, 1), **kwargs):
        from statsmodels.tsa.arima.model import ARIMA

        self._model = ARIMA(series, order=order).fit()

    def _fit_exp_smoothing(self, series: pd.Series, **kwargs):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        seasonal_periods = kwargs.get("seasonal_periods", 12)
        self._model = ExponentialSmoothing(
            series, trend="add", seasonal="add", seasonal_periods=seasonal_periods
        ).fit()

    def _fit_rolling_mean(self, series: pd.Series, window: int = 12):
        self._last_rolling_mean = series.rolling(window).mean().iloc[-1]
