"""Effect size and power analysis helpers.

This module provides :class:`EffectSizeCalculator` with commonly used effect
size metrics and simple power/sample-size approximations.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


class EffectSizeCalculator:
    """Compute effect sizes for independent tests and association tables."""

    @staticmethod
    def cohens_d(group_a, group_b) -> dict:
        """Compute Cohen's d for two independent groups.

        Returns both numeric effect and qualitative magnitude label.
        """
        na, nb = len(group_a), len(group_b)
        ma, mb = np.mean(group_a), np.mean(group_b)
        va, vb = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
        pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
        d = (ma - mb) / pooled_std if pooled_std > 0 else 0.0
        magnitude = (
            "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        )
        return {"cohens_d": d, "magnitude": magnitude}

    @staticmethod
    def cramers_v(contingency_table) -> dict:
        """Compute Cramér's V from a contingency table."""
        chi2 = stats.chi2_contingency(contingency_table)[0]
        n = (
            np.sum(contingency_table.values)
            if hasattr(contingency_table, "values")
            else np.sum(contingency_table)
        )
        min_dim = min(contingency_table.shape) - 1
        v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0
        magnitude = (
            "negligible" if v < 0.1 else "small" if v < 0.3 else "medium" if v < 0.5 else "large"
        )
        return {"cramers_v": v, "magnitude": magnitude}

    @staticmethod
    def eta_squared(*args) -> dict:
        """Compute eta-squared effect size.

        Supported call signatures:

        - ``eta_squared(f_statistic, df_between, df_within)``
        - ``eta_squared(*groups)`` (computes ANOVA internally)
        """
        if len(args) == 3 and all(np.isscalar(a) for a in args):
            f_statistic, df_between, df_within = args
        else:
            f_statistic, _ = stats.f_oneway(*args)
            df_between = len(args) - 1
            df_within = sum(len(g) for g in args) - len(args)

        eta2 = (f_statistic * df_between) / (f_statistic * df_between + df_within)
        magnitude = (
            "negligible" if eta2 < 0.01 else "small" if eta2 < 0.06 else "medium" if eta2 < 0.14 else "large"
        )
        return {"eta_squared": eta2, "magnitude": magnitude}

    @staticmethod
    def power_analysis(
        effect_size: float,
        n: int | None = None,
        alpha: float = 0.05,
        power: float | None = None,
    ) -> dict:
        """Approximate power/sample-size relationship for two-sample t-test.

        Provide one of:

        - ``n`` to estimate achieved statistical power, or
        - ``power`` to estimate required sample size per group.

        Raises
        ------
        ValueError
            If neither or both of the required modes are specified.
        """
        if n is not None and power is None:
            se = np.sqrt(2 / n)
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_power = (effect_size / se) - z_alpha
            computed_power = stats.norm.cdf(z_power)
            return {
                "power": computed_power,
                "effect_size": effect_size,
                "n": n,
                "alpha": alpha,
            }
        if power is not None and n is None:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            z_beta = stats.norm.ppf(power)
            n_required = int(np.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2))
            return {
                "required_n_per_group": n_required,
                "effect_size": effect_size,
                "alpha": alpha,
                "target_power": power,
            }
        raise ValueError("Specify exactly one mode: either n or power")
