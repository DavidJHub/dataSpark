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

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def fit_distributions(
        self, column: str, candidates: list | None = None
    ) -> pd.DataFrame:
        """Fit candidate distributions via MLE; rank by BIC / KS-test."""
        data = self.df[column].dropna().values
        dists = candidates or CANDIDATE_DISTRIBUTIONS
        results = []
        for dist in dists:
            try:
                params = dist.fit(data)
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(data, dist.cdf, args=params)
                # Log-likelihood & BIC
                ll = np.sum(dist.logpdf(data, *params))
                k = len(params)
                n = len(data)
                bic = k * np.log(n) - 2 * ll
                results.append({
                    "distribution": dist.name,
                    "params": params,
                    "ks_statistic": ks_stat,
                    "ks_p_value": ks_p,
                    "log_likelihood": ll,
                    "bic": bic,
                })
            except Exception as e:
                logger.debug("Fit failed for {} — {}", dist.name, e)
        return pd.DataFrame(results).sort_values("bic").reset_index(drop=True)

    def detect_multimodality(self, column: str) -> dict:
        """Hartigan's dip test approximation for multimodality."""
        data = np.sort(self.df[column].dropna().values)
        n = len(data)
        # Simple bimodality coefficient (BC)
        skew = stats.skew(data)
        kurt = stats.kurtosis(data, fisher=True)
        bc = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        return {
            "column": column,
            "bimodality_coefficient": bc,
            "likely_multimodal": bc > 0.555,
            "skewness": skew,
            "excess_kurtosis": kurt,
        }

    def quantile_analysis(self, column: str, quantiles: list[float] | None = None) -> dict:
        """Detailed quantile analysis."""
        q = quantiles or [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        data = self.df[column].dropna()
        return {
            "column": column,
            "n": len(data),
            "quantiles": {f"q{int(qi*100):02d}": data.quantile(qi) for qi in q},
            "iqr": data.quantile(0.75) - data.quantile(0.25),
            "range": data.max() - data.min(),
        }
