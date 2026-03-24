"""
Time Series Feature Extraction
===============================
Extract statistical features from time windows for ML-based forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class TimeSeriesFeatureExtractor:
    """Extract summary features from time-series windows."""

    @staticmethod
    def rolling_features(
        data: pd.Series | pd.DataFrame,
        windows: list[int] | None = None,
        *,
        column: str | None = None,
    ) -> pd.DataFrame:
        """Compute rolling statistics for multiple window sizes.

        Accepts a Series or a DataFrame with a column parameter.
        """
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = data.select_dtypes(include="number").columns[0]
            series = data[column]
        else:
            series = data
        windows = windows or [7, 14, 30, 90]
        df = pd.DataFrame(index=series.index)
        for w in windows:
            df[f"rolling_mean_{w}"] = series.rolling(w).mean()
            df[f"rolling_std_{w}"] = series.rolling(w).std()
            df[f"rolling_min_{w}"] = series.rolling(w).min()
            df[f"rolling_max_{w}"] = series.rolling(w).max()
            df[f"rolling_median_{w}"] = series.rolling(w).median()
        return df

    @staticmethod
    def lag_features(
        data: pd.Series | pd.DataFrame,
        lags: list[int] | None = None,
        *,
        column: str | None = None,
    ) -> pd.DataFrame:
        """Create lag features.

        Accepts a Series or a DataFrame with a column parameter.
        """
        if isinstance(data, pd.DataFrame):
            if column is None:
                column = data.select_dtypes(include="number").columns[0]
            series = data[column]
        else:
            series = data
        lags = lags or [1, 2, 3, 7, 14, 30]
        df = pd.DataFrame(index=series.index)
        for lag in lags:
            df[f"lag_{lag}"] = series.shift(lag)
        return df

    @staticmethod
    def datetime_features(data: pd.DatetimeIndex | pd.DataFrame) -> pd.DataFrame:
        """Extract calendar features from a DatetimeIndex or DataFrame with datetime index."""
        if isinstance(data, pd.DataFrame):
            index = data.index
        else:
            index = data
        if not isinstance(index, pd.DatetimeIndex):
            index = pd.DatetimeIndex(index)
        return pd.DataFrame({
            "year": index.year,
            "month": index.month,
            "day": index.day,
            "day_of_week": index.dayofweek,
            "day_of_year": index.dayofyear,
            "week_of_year": index.isocalendar().week.values,
            "quarter": index.quarter,
            "is_weekend": index.dayofweek >= 5,
            "is_month_start": index.is_month_start,
            "is_month_end": index.is_month_end,
        }, index=index)

    @staticmethod
    def window_statistics(series: pd.Series, window: int = 30) -> pd.DataFrame:
        """Advanced windowed statistics: skew, kurtosis, entropy."""
        df = pd.DataFrame(index=series.index)
        df["win_skew"] = series.rolling(window).skew()
        df["win_kurtosis"] = series.rolling(window).kurt()
        df["win_range"] = series.rolling(window).max() - series.rolling(window).min()
        df["win_cv"] = series.rolling(window).std() / series.rolling(window).mean()
        df["pct_change_1"] = series.pct_change(1)
        df["pct_change_7"] = series.pct_change(7)
        return df
