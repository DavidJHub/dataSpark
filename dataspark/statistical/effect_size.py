"""
Effect Size Calculators
=======================
Cohen's d, Cramér's V, eta-squared, and more.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


class EffectSizeCalculator:
    """Compute effect sizes for various test scenarios."""

    @staticmethod
    def cohens_d(group_a, group_b) -> dict:
        """Cohen's d for two independent groups."""
        na, nb = len(group_a), len(group_b)
        ma, mb = np.mean(group_a), np.mean(group_b)
        va, vb = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
        pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
        d = (ma - mb) / pooled_std if pooled_std > 0 else 0.0
        magnitude = (
            "negligible" if abs(d) < 0.2
            else "small" if abs(d) < 0.5
            else "medium" if abs(d) < 0.8
            else "large"
        )
        return {"cohens_d": d, "magnitude": magnitude}

    @staticmethod
    def cramers_v(contingency_table) -> dict:
        """Cramér's V for chi-squared test of association."""
        chi2 = stats.chi2_contingency(contingency_table)[0]
        n = np.sum(contingency_table.values) if hasattr(contingency_table, 'values') else np.sum(contingency_table)
        min_dim = min(contingency_table.shape) - 1
        v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0
        return {"cramers_v": v}

    @staticmethod
    def eta_squared(f_statistic: float, df_between: int, df_within: int) -> dict:
        """Eta-squared from ANOVA F-statistic."""
        eta2 = (f_statistic * df_between) / (f_statistic * df_between + df_within)
        return {"eta_squared": eta2}

    @staticmethod
    def power_analysis(effect_size: float, n: int, alpha: float = 0.05) -> dict:
        """Approximate statistical power for a two-sample t-test."""
        se = np.sqrt(2 / n)
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_power = (effect_size / se) - z_alpha
        power = stats.norm.cdf(z_power)
        return {"power": power, "effect_size": effect_size, "n": n, "alpha": alpha}
