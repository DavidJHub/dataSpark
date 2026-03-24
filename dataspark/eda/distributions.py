"""Statistical distribution fitting and shape diagnostics.

This module provides :class:`DistributionAnalyzer` to fit candidate
parametric distributions, evaluate goodness-of-fit, inspect multimodality,
and summarize quantiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


CANDIDATE_DISTRIBUTIONS = [
    stats.norm,
    stats.lognorm,
    stats.expon,
    stats.gamma,
    stats.beta,
    stats.weibull_min,
    stats.uniform,
]


class DistributionAnalyzer:
    """Analyze and fit statistical distributions to numeric data.

    Parameters
    ----------
    df:
        Optional dataframe used by legacy column-based APIs.
    """

    def __init__(self, df: pd.DataFrame | None = None) -> None:
        """Initialize analyzer with optional dataframe context."""
        self.df = df

    def fit(self, data: pd.Series | np.ndarray, candidates: list | None = None) -> pd.DataFrame:
        """Fit candidate distributions using maximum likelihood estimation.

        For each candidate distribution this method computes:

        - fitted parameters,
        - Kolmogorov-Smirnov statistic and p-value,
        - log-likelihood,
        - BIC (Bayesian Information Criterion).

        Parameters
        ----------
        data:
            Numeric input data (series or array).
        candidates:
            Optional list of SciPy distributions to evaluate. If omitted,
            :data:`CANDIDATE_DISTRIBUTIONS` is used.

        Returns
        -------
        pandas.DataFrame
            Ranked fit results sorted by increasing BIC.
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
                results.append(
                    {
                        "distribution": dist.name,
                        "params": params,
                        "ks_statistic": ks_stat,
                        "ks_pvalue": ks_p,
                        "log_likelihood": ll,
                        "bic": bic,
                    }
                )
            except Exception as e:
                logger.debug("Fit failed for {} — {}", dist.name, e)
        return pd.DataFrame(results).sort_values("bic").reset_index(drop=True)

    def fit_distributions(self, column: str, candidates: list | None = None) -> pd.DataFrame:
        """Legacy wrapper: fit candidate distributions for ``self.df[column]``.

        Raises
        ------
        ValueError
            If no dataframe was provided at initialization.
        """
        if self.df is None:
            raise ValueError("DataFrame not set. Pass df to __init__ or use fit() directly.")
        return self.fit(self.df[column].dropna(), candidates)

    def multimodality(self, data: pd.Series | np.ndarray) -> dict:
        """Estimate multimodality using the bimodality coefficient.

        Parameters
        ----------
        data:
            Numeric input data.

        Returns
        -------
        dict
            Includes bimodality coefficient, boolean flag, skewness, and
            excess kurtosis.
        """
        if isinstance(data, pd.Series):
            data = data.dropna().values
        data = np.sort(np.asarray(data, dtype=float))
        n = len(data)
        skew = stats.skew(data)
        kurt = stats.kurtosis(data, fisher=True)
        bc = (skew**2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        return {
            "bimodality_coefficient": bc,
            "is_multimodal": bc > 0.555,
            "skewness": skew,
            "excess_kurtosis": kurt,
        }

    def detect_multimodality(self, column: str) -> dict:
        """Legacy wrapper: run multimodality check for ``self.df[column]``."""
        if self.df is None:
            raise ValueError(
                "DataFrame not set. Pass df to __init__ or use multimodality() directly."
            )
        return self.multimodality(self.df[column].dropna())

    def quantile_analysis(self, column: str, quantiles: list[float] | None = None) -> dict:
        """Compute quantiles, IQR and range for a dataframe column.

        Parameters
        ----------
        column:
            Target numeric column from ``self.df``.
        quantiles:
            Optional quantile list in [0, 1]. If omitted, common percentiles are
            used.
        """
        if self.df is None:
            raise ValueError("DataFrame not set. Pass df to __init__.")
        q = quantiles or [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        data = self.df[column].dropna()
        return {
            "column": column,
            "n": len(data),
            "quantiles": {f"q{int(qi * 100):02d}": data.quantile(qi) for qi in q},
            "iqr": data.quantile(0.75) - data.quantile(0.25),
            "range": data.max() - data.min(),
        }
