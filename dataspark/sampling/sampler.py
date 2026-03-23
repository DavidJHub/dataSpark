"""
Sampling Module
===============
Stratified, cluster, systematic, reservoir, and bootstrap sampling
for survey and ML applications.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger


class Sampler:
    """Statistical sampling methods for DataFrames."""

    def __init__(self, random_state: int = 42) -> None:
        self.rng = np.random.default_rng(random_state)

    def stratified_sample(
        self,
        df: pd.DataFrame,
        stratum_col: str,
        n: int | None = None,
        frac: float | None = None,
    ) -> pd.DataFrame:
        """Proportional stratified sampling."""
        if n is not None:
            # proportional allocation
            counts = df[stratum_col].value_counts(normalize=True)
            samples = []
            for stratum, prop in counts.items():
                stratum_df = df[df[stratum_col] == stratum]
                k = max(1, int(round(n * prop)))
                k = min(k, len(stratum_df))
                samples.append(stratum_df.sample(n=k, random_state=self.rng.integers(1e9)))
            result = pd.concat(samples).reset_index(drop=True)
        elif frac is not None:
            samples = []
            for stratum in df[stratum_col].unique():
                stratum_df = df[df[stratum_col] == stratum]
                samples.append(
                    stratum_df.sample(frac=frac, random_state=self.rng.integers(1e9))
                )
            result = pd.concat(samples).reset_index(drop=True)
        else:
            raise ValueError("Must specify either n or frac")
        logger.info("Stratified sample: {} → {} rows", len(df), len(result))
        return result

    def cluster_sample(
        self,
        df: pd.DataFrame,
        cluster_col: str,
        n_clusters: int,
    ) -> pd.DataFrame:
        """Select n random clusters and include all their observations."""
        clusters = df[cluster_col].unique()
        selected = self.rng.choice(clusters, size=min(n_clusters, len(clusters)), replace=False)
        result = df[df[cluster_col].isin(selected)].reset_index(drop=True)
        logger.info("Cluster sample: {} clusters, {} rows", len(selected), len(result))
        return result

    def systematic_sample(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        """Every k-th element with a random start."""
        start = self.rng.integers(0, k)
        indices = np.arange(start, len(df), k)
        result = df.iloc[indices].reset_index(drop=True)
        logger.info("Systematic sample (k={}): {} rows", k, len(result))
        return result

    def reservoir_sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Reservoir sampling (Vitter's Algorithm R) for streaming scenarios."""
        reservoir = df.iloc[:n].copy()
        for i in range(n, len(df)):
            j = self.rng.integers(0, i + 1)
            if j < n:
                reservoir.iloc[j] = df.iloc[i]
        logger.info("Reservoir sample: {} rows", n)
        return reservoir.reset_index(drop=True)

    def bootstrap_sample(
        self,
        df: pd.DataFrame,
        n_samples: int = 1000,
        statistic: str = "mean",
        column: str | None = None,
    ) -> dict:
        """Bootstrap resampling for confidence intervals."""
        col = column or df.select_dtypes(include="number").columns[0]
        stat_fn = getattr(np, statistic)
        boot_stats = []
        for _ in range(n_samples):
            sample = df[col].sample(n=len(df), replace=True, random_state=self.rng.integers(1e9))
            boot_stats.append(stat_fn(sample))
        boot_stats = np.array(boot_stats)
        return {
            "statistic": statistic,
            "column": col,
            "mean": float(np.mean(boot_stats)),
            "std": float(np.std(boot_stats)),
            "ci_95_lower": float(np.percentile(boot_stats, 2.5)),
            "ci_95_upper": float(np.percentile(boot_stats, 97.5)),
            "n_samples": n_samples,
        }

    def sample_size_calculator(
        self,
        population_size: int,
        confidence_level: float = 0.95,
        margin_of_error: float = 0.05,
        proportion: float = 0.5,
    ) -> dict:
        """Calculate required sample size using Cochran's formula."""
        from scipy import stats as sp_stats

        z = sp_stats.norm.ppf(1 - (1 - confidence_level) / 2)
        n0 = (z ** 2 * proportion * (1 - proportion)) / (margin_of_error ** 2)
        # Finite population correction
        n = n0 / (1 + (n0 - 1) / population_size)
        return {
            "required_sample_size": int(np.ceil(n)),
            "population_size": population_size,
            "confidence_level": confidence_level,
            "margin_of_error": margin_of_error,
            "cochran_n0": int(np.ceil(n0)),
        }
