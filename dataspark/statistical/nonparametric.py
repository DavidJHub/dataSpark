"""Non-parametric hypothesis testing utilities.

This module provides distribution-free alternatives to parametric tests via
:class:`NonParametricTests`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class NonParametricTests:
    """Non-parametric hypothesis tests with structured outputs."""

    @staticmethod
    def mann_whitney(group_a, group_b, alternative: str = "two-sided") -> dict:
        """Run Mann-Whitney U test for two independent samples.

        Also reports rank-biserial effect size approximation.
        """
        stat, p = stats.mannwhitneyu(group_a, group_b, alternative=alternative)
        n1, n2 = len(group_a), len(group_b)
        r = 1 - (2 * stat) / (n1 * n2)
        return {
            "test": "mann_whitney_u",
            "statistic": stat,
            "p_value": p,
            "effect_size_r": r,
            "alternative": alternative,
        }

    @staticmethod
    def wilcoxon_signed_rank(x, y=None, alternative: str = "two-sided") -> dict:
        """Run Wilcoxon signed-rank test for paired/one-sample settings."""
        stat, p = stats.wilcoxon(x, y, alternative=alternative)
        return {"test": "wilcoxon_signed_rank", "statistic": stat, "p_value": p}

    @staticmethod
    def kruskal_wallis(*groups) -> dict:
        """Run Kruskal-Wallis H test (non-parametric one-way ANOVA)."""
        stat, p = stats.kruskal(*groups)
        return {
            "test": "kruskal_wallis",
            "statistic": stat,
            "p_value": p,
            "n_groups": len(groups),
        }

    @staticmethod
    def ks_two_sample(a, b) -> dict:
        """Run two-sample Kolmogorov-Smirnov test."""
        stat, p = stats.ks_2samp(a, b)
        return {"test": "ks_two_sample", "statistic": stat, "p_value": p}

    @staticmethod
    def friedman(*groups) -> dict:
        """Run Friedman test for repeated measures with >=3 conditions."""
        stat, p = stats.friedmanchisquare(*groups)
        return {"test": "friedman", "statistic": stat, "p_value": p, "n_groups": len(groups)}

    @staticmethod
    def runs_test(series: pd.Series) -> dict:
        """Run Wald-Wolfowitz runs test for randomness around median."""
        median = series.median()
        binary = (series >= median).astype(int).values
        runs = 1 + np.sum(np.diff(binary) != 0)
        n1 = np.sum(binary == 1)
        n0 = np.sum(binary == 0)
        n = n1 + n0
        expected = 1 + (2 * n1 * n0) / n
        var = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n**2 * (n - 1)) if n > 1 else 1
        z = (runs - expected) / np.sqrt(var) if var > 0 else 0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return {
            "test": "runs_test",
            "n_runs": int(runs),
            "expected_runs": expected,
            "z_statistic": z,
            "p_value": p,
        }
