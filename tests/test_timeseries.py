"""Tests for dataspark.timeseries module."""

import numpy as np
import pandas as pd
import pytest

from dataspark.timeseries import TimeSeriesDecomposer, Forecaster, TimeSeriesFeatureExtractor


class TestTimeSeriesDecomposer:
    def test_stl_decompose(self, time_series):
        decomposer = TimeSeriesDecomposer(method="stl", period=30)
        result = decomposer.decompose(time_series)
        assert "trend" in result
        assert "seasonal" in result
        assert "residual" in result

    def test_classical_decompose(self, time_series):
        decomposer = TimeSeriesDecomposer(method="classical", period=30)
        result = decomposer.decompose(time_series)
        assert "trend" in result

    def test_trend_test(self, time_series):
        decomposer = TimeSeriesDecomposer()
        result = decomposer.trend_test(time_series)
        assert result["trend"] == "increasing"  # we built an upward trend

    def test_stationarity_test(self, time_series):
        decomposer = TimeSeriesDecomposer()
        result = decomposer.stationarity_test(time_series)
        assert "is_stationary" in result
        assert "critical_values" in result


class TestForecaster:
    def test_arima(self, time_series):
        forecaster = Forecaster(method="arima")
        forecaster.fit(time_series, order=(1, 1, 0))
        preds = forecaster.predict(steps=10)
        assert len(preds) == 10

    def test_evaluate(self, time_series):
        train = time_series[:300]
        test = time_series[300:]
        forecaster = Forecaster(method="arima")
        metrics = forecaster.evaluate(train, test)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["rmse"] >= 0


class TestTimeSeriesFeatureExtractor:
    def test_rolling_features(self, time_series):
        result = TimeSeriesFeatureExtractor.rolling_features(time_series, windows=[7, 14])
        assert "rolling_mean_7" in result.columns
        assert "rolling_std_14" in result.columns

    def test_lag_features(self, time_series):
        result = TimeSeriesFeatureExtractor.lag_features(time_series, lags=[1, 7])
        assert "lag_1" in result.columns
        assert "lag_7" in result.columns

    def test_datetime_features(self, time_series):
        result = TimeSeriesFeatureExtractor.datetime_features(time_series.index)
        assert "month" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns

    def test_window_statistics(self, time_series):
        result = TimeSeriesFeatureExtractor.window_statistics(time_series, window=14)
        assert "win_skew" in result.columns
        assert "pct_change_1" in result.columns
