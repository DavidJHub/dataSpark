"""Parametric hypothesis testing utilities.

This module groups common parametric significance tests in
:class:`HypothesisTester` and returns structured dictionary outputs suitable
for reporting pipelines and dashboards.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


class HypothesisTester:
    """Parametric hypothesis tests with standardized result dictionaries."""

    @staticmethod
    def t_test(
        group_a: np.ndarray | pd.Series,
        group_b: np.ndarray | pd.Series,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        equal_var: bool = False,
    ) -> dict:
        """Run independent-samples t-test.

        Parameters
        ----------
        group_a, group_b:
            Numeric samples from two independent groups.
        alternative:
            Alternative hypothesis for the test.
        equal_var:
            If ``False`` (default), Welch's t-test is used. If ``True``,
            Student's t-test with equal variances is used.

        Returns
        -------
        dict
            Test name, statistic, p-value, direction, sample means and sizes.
        """
        stat, p = stats.ttest_ind(group_a, group_b, equal_var=equal_var, alternative=alternative)
        return {
            "test": "Welch's t-test" if not equal_var else "Student's t-test",
            "statistic": stat,
            "p_value": p,
            "alternative": alternative,
            "mean_a": float(np.nanmean(group_a)),
            "mean_b": float(np.nanmean(group_b)),
            "n_a": len(group_a),
            "n_b": len(group_b),
        }

    @staticmethod
    def paired_t_test(
        before: np.ndarray | pd.Series,
        after: np.ndarray | pd.Series,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> dict:
        """Run paired-samples t-test for repeated measurements.

        Parameters
        ----------
        before, after:
            Aligned measurements from the same entities at two time points.
        alternative:
            Alternative hypothesis for paired difference.
        """
        stat, p = stats.ttest_rel(before, after, alternative=alternative)
        diff = np.asarray(after) - np.asarray(before)
        return {
            "test": "paired_t_test",
            "statistic": stat,
            "p_value": p,
            "mean_diff": float(np.nanmean(diff)),
            "std_diff": float(np.nanstd(diff, ddof=1)),
        }

    @staticmethod
    def one_way_anova(*groups: np.ndarray | pd.Series) -> dict:
        """Run one-way ANOVA F-test across multiple groups.

        Parameters
        ----------
        *groups:
            Two or more numeric group arrays/series.
        """
        stat, p = stats.f_oneway(*groups)
        return {
            "test": "one_way_anova",
            "statistic": stat,
            "p_value": p,
            "n_groups": len(groups),
            "group_sizes": [len(g) for g in groups],
            "group_means": [float(np.nanmean(g)) for g in groups],
        }

    anova = one_way_anova

    @staticmethod
    def chi_squared(observed: pd.DataFrame) -> dict:
        """Run chi-squared test of independence on contingency table.

        Parameters
        ----------
        observed:
            Contingency table of observed counts.
        """
        chi2, p, dof, expected = stats.chi2_contingency(observed)
        return {
            "test": "chi_squared",
            "statistic": chi2,
            "p_value": p,
            "degrees_of_freedom": dof,
            "expected_frequencies": expected,
        }

    @staticmethod
    def proportion_z_test(successes_a: int, n_a: int, successes_b: int, n_b: int) -> dict:
        """Run two-proportion z-test.

        Parameters
        ----------
        successes_a, n_a:
            Success count and sample size for group A.
        successes_b, n_b:
            Success count and sample size for group B.
        """
        p1 = successes_a / n_a
        p2 = successes_b / n_b
        p_pool = (successes_a + successes_b) / (n_a + n_b)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
        z = (p1 - p2) / se if se > 0 else 0.0
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        return {
            "test": "two_proportion_z",
            "z_statistic": z,
            "p_value": p_val,
            "prop_a": p1,
            "prop_b": p2,
            "pooled_prop": p_pool,
        }
