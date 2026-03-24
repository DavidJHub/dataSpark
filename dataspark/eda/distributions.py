"""
Distribution Analysis
=====================
Fit parametric distributions, compare goodness-of-fit,
and detect multimodality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


CANDIDATE_DISTRIBUTIONS = [
    stats.norm, stats.lognorm, stats.expon, stats.gamma,
    stats.beta, stats.weibull_min, stats.uniform,
]


class DistributionAnalyzer:
    """Analyze and fit statistical distributions to numeric data."""

    def __init__(self, df: pd.DataFrame | None = None) -> None:
        self.df = df

    def fit(
        self, data: pd.Series | np.ndarray, candidates: list | None = None
    ) -> pd.DataFrame:
        """Fit candidate distributions via MLE; rank by BIC / KS-test.

        Accepts raw data (Series or array) directly.
        """
        if isinstance(data, pd.Series):
            data = data.dropna().values
        data = np.asarray(data, dtype=float)
        dists = candidates or CANDIDATE_DISTRIBUTIONS
        results = []
        for dist in dists:
            try:
                params = dist.fit(data)
                ks_stat, ks_p = stats.kstest(data, dist.cdf, args=params)
                ll = np.sum(dist.logpdf(data, *params))
                k = len(params)
                n = len(data)
                bic = k * np.log(n) - 2 * ll
                results.append({
                    "distribution": dist.name,
                    "params": params,
                    "ks_statistic": ks_stat,
                    "ks_pvalue": ks_p,
                    "log_likelihood": ll,
                    "bic": bic,
                })
            except Exception as e:
                logger.debug("Fit failed for {} — {}", dist.name, e)
        return pd.DataFrame(results).sort_values("bic").reset_index(drop=True)

    def fit_distributions(
        self, column: str, candidates: list | None = None
    ) -> pd.DataFrame:
        """Fit distributions using a column name from self.df (legacy API)."""
        if self.df is None:
            raise ValueError("DataFrame not set. Pass df to __init__ or use fit() directly.")
        return self.fit(self.df[column].dropna(), candidates)

    def multimodality(self, data: pd.Series | np.ndarray) -> dict:
        """Check for multimodality. Accepts raw data directly."""
        if isinstance(data, pd.Series):
            data = data.dropna().values
        data = np.sort(np.asarray(data, dtype=float))
        n = len(data)
        skew = stats.skew(data)
        kurt = stats.kurtosis(data, fisher=True)
        bc = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        return {
            "bimodality_coefficient": bc,
            "is_multimodal": bc > 0.555,
            "skewness": skew,
            "excess_kurtosis": kurt,
        }

    def detect_multimodality(self, column: str) -> dict:
        """Detect multimodality using a column name from self.df (legacy API)."""
        if self.df is None:
            raise ValueError("DataFrame not set. Pass df to __init__ or use multimodality() directly.")
        return self.multimodality(self.df[column].dropna())

    def quantile_analysis(self, column: str, quantiles: list[float] | None = None) -> dict:
        """Detailed quantile analysis."""
        if self.df is None:
            raise ValueError("DataFrame not set. Pass df to __init__.")
        q = quantiles or [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        data = self.df[column].dropna()
        return {
            "column": column,
            "n": len(data),
            "quantiles": {f"q{int(qi*100):02d}": data.quantile(qi) for qi in q},
            "iqr": data.quantile(0.75) - data.quantile(0.25),
            "range": data.max() - data.min(),
        }
