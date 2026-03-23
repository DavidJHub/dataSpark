"""
Time Series Decomposition
=========================
Trend-cycle, seasonal, and residual decomposition
using classical, STL, and rolling-window approaches.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from scipy import stats
from loguru import logger


class TimeSeriesDecomposer:
    """Decompose time series into trend, seasonal, and residual components."""

    def __init__(
        self,
        method: Literal["classical", "stl"] = "stl",
        period: int | None = None,
    ) -> None:
        self.method = method
        self.period = period

    def decompose(
        self, series: pd.Series, model: Literal["additive", "multiplicative"] = "additive"
    ) -> dict:
        """Return decomposition components as a dict of Series."""
        series = series.dropna()
        period = self.period or self._estimate_period(series)

        if self.method == "stl":
            result = STL(series, period=period).fit()
        else:
            result = seasonal_decompose(series, model=model, period=period)

        logger.info("Decomposed series (method={}, period={})", self.method, period)
        return {
            "observed": result.observed,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
            "period": period,
        }

    def trend_test(self, series: pd.Series) -> dict:
        """Mann-Kendall trend test."""
        data = series.dropna().values
        n = len(data)
        s = 0
        for k in range(n - 1):
            for j in range(k + 1, n):
                s += np.sign(data[j] - data[k])
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return {
            "test": "mann_kendall",
            "s_statistic": s,
            "z_statistic": z,
            "p_value": p,
            "trend": "increasing" if z > 0 and p < 0.05 else "decreasing" if z < 0 and p < 0.05 else "no_trend",
        }

    def stationarity_test(self, series: pd.Series) -> dict:
        """Augmented Dickey-Fuller test for stationarity."""
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(series.dropna(), autolag="AIC")
        return {
            "test": "adf",
            "statistic": result[0],
            "p_value": result[1],
            "lags_used": result[2],
            "n_obs": result[3],
            "critical_values": result[4],
            "is_stationary": result[1] < 0.05,
        }

    @staticmethod
    def _estimate_period(series: pd.Series) -> int:
        """Estimate dominant period via autocorrelation."""
        from statsmodels.tsa.stattools import acf

        n = len(series)
        nlags = min(n // 2, 400)
        autocorr = acf(series, nlags=nlags, fft=True)
        # Find first peak after lag 1
        for i in range(2, len(autocorr) - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                return i
        return 12  # default fallback
