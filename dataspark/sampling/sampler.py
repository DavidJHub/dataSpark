"""Statistical sampling utilities for tabular datasets.

This module exposes :class:`Sampler`, a compact utility class with common
sampling strategies used in survey design, EDA prototyping, model development,
and uncertainty estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class Sampler:
    """Collection of statistical sampling methods for DataFrames.

    Parameters
    ----------
    random_state:
        Seed used to initialize NumPy's random generator for reproducible
        sampling operations.
    """

    def __init__(self, random_state: int = 42) -> None:
        """Create a sampler instance with deterministic random generator."""
        self.rng = np.random.default_rng(random_state)

    def stratified_sample(
        self,
        df: pd.DataFrame,
        stratum_col: str,
        n: int | None = None,
        frac: float | None = None,
    ) -> pd.DataFrame:
        """Draw a stratified sample with proportional allocation.

        Parameters
        ----------
        df:
            Input dataframe.
        stratum_col:
            Column defining stratum membership.
        n:
            Total target sample size. When provided, each stratum receives an
            approximate proportional quota and at least one record.
        frac:
            Fraction sampled inside each stratum.

        Returns
        -------
        pandas.DataFrame
            Stratified sample as a new dataframe.

        Raises
        ------
        ValueError
            If neither ``n`` nor ``frac`` is provided.
        """
        if n is not None:
            counts = df[stratum_col].value_counts(normalize=True)
            samples = []
            for stratum, prop in counts.items():
                stratum_df = df[df[stratum_col] == stratum]
                k = max(1, int(round(n * prop)))
                k = min(k, len(stratum_df))
                samples.append(stratum_df.sample(n=k, random_state=self.rng.integers(1e9)))
            result = pd.concat(samples).reset_index(drop=True)
        elif frac is not None:
            result = (
                df.groupby(stratum_col, group_keys=False)
                .apply(lambda x: x.sample(frac=frac, random_state=self.rng.integers(1e9)))
                .reset_index(drop=True)
            )
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
        """Sample complete clusters and include all observations in them.

        Parameters
        ----------
        df:
            Input dataframe.
        cluster_col:
            Column containing cluster labels.
        n_clusters:
            Number of clusters to select without replacement.

        Returns
        -------
        pandas.DataFrame
            Dataframe containing all rows from the selected clusters.
        """
        clusters = df[cluster_col].unique()
        selected = self.rng.choice(clusters, size=min(n_clusters, len(clusters)), replace=False)
        result = df[df[cluster_col].isin(selected)].reset_index(drop=True)
        logger.info("Cluster sample: {} clusters, {} rows", len(selected), len(result))
        return result

    def systematic_sample(
        self,
        df: pd.DataFrame,
        k: int | None = None,
        *,
        n: int | None = None,
    ) -> pd.DataFrame:
        """Select every ``k``-th observation using a random start offset.

        Parameters
        ----------
        df:
            Input dataframe.
        k:
            Sampling interval.
        n:
            Desired approximate sample size. When provided and ``k`` is not,
            ``k`` is computed as ``len(df) // n`` with lower bound 1.

        Returns
        -------
        pandas.DataFrame
            Systematic sample as a new dataframe.

        Raises
        ------
        ValueError
            If both ``k`` and ``n`` are omitted.
        """
        if n is not None and k is None:
            k = max(1, len(df) // n)
        elif k is None:
            raise ValueError("Must specify either k or n")
        start = self.rng.integers(0, k)
        indices = np.arange(start, len(df), k)
        result = df.iloc[indices].reset_index(drop=True)
        logger.info("Systematic sample (k={}): {} rows", k, len(result))
        return result

    def reservoir_sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Perform reservoir sampling (Algorithm R) over dataframe rows.

        Parameters
        ----------
        df:
            Input dataframe; can represent static or stream-like row iteration.
        n:
            Reservoir size (target sample size).

        Returns
        -------
        pandas.DataFrame
            Uniform random sample of size ``n`` (or ``len(df)`` when ``n``
            exceeds available rows).
        """
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
        """Estimate statistic distribution via bootstrap resampling.

        Parameters
        ----------
        df:
            Input dataframe.
        n_samples:
            Number of bootstrap replicates.
        statistic:
            Name of NumPy reduction function (e.g. ``"mean"``, ``"median"``,
            ``"std"``) applied to each replicate.
        column:
            Numeric column to analyze. If omitted, the first numeric column is
            used.

        Returns
        -------
        dict
            Dictionary with statistic summary, bootstrap standard deviation,
            and percentile confidence interval (2.5%, 97.5%).
        """
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
            "statistic_value": float(np.mean(boot_stats)),
            "std": float(np.std(boot_stats)),
            "ci_lower": float(np.percentile(boot_stats, 2.5)),
            "ci_upper": float(np.percentile(boot_stats, 97.5)),
            "n_samples": n_samples,
        }

    def sample_size_calculator(
        self,
        population_size: int,
        confidence_level: float = 0.95,
        margin_of_error: float = 0.05,
        proportion: float = 0.5,
    ) -> dict:
        """Compute required sample size using Cochran's formula.

        Applies finite population correction (FPC) to adapt the infinite
        population estimate ``n0`` to a finite ``population_size``.

        Parameters
        ----------
        population_size:
            Size of the target population.
        confidence_level:
            Desired confidence level (e.g. 0.95).
        margin_of_error:
            Maximum tolerated estimation error.
        proportion:
            Expected population proportion (use 0.5 for conservative estimate).

        Returns
        -------
        dict
            Summary with corrected required sample size and intermediate values.
        """
        from scipy import stats as sp_stats

        z = sp_stats.norm.ppf(1 - (1 - confidence_level) / 2)
        n0 = (z**2 * proportion * (1 - proportion)) / (margin_of_error**2)
        n = n0 / (1 + (n0 - 1) / population_size)
        return {
            "required_sample_size": int(np.ceil(n)),
            "population_size": population_size,
            "confidence_level": confidence_level,
            "margin_of_error": margin_of_error,
            "cochran_n0": int(np.ceil(n0)),
        }
