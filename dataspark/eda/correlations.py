"""
Correlation Analysis
====================
Pearson, Spearman, Kendall, and point-biserial correlations
with statistical significance testing.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


class CorrelationAnalyzer:
    """Compute and analyze correlations with significance tests."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.numeric = df.select_dtypes(include="number")

    def correlation_matrix(
        self, method: Literal["pearson", "spearman", "kendall"] = "pearson"
    ) -> pd.DataFrame:
        """Standard correlation matrix."""
        return self.numeric.corr(method=method)

    def pairwise_significance(
        self, method: Literal["pearson", "spearman"] = "pearson", alpha: float = 0.05
    ) -> pd.DataFrame:
        """Pairwise correlations with p-values and significance flags."""
        cols = self.numeric.columns
        rows = []
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                data = self.df[[c1, c2]].dropna()
                if len(data) < 3:
                    continue
                if method == "pearson":
                    r, p = stats.pearsonr(data[c1], data[c2])
                else:
                    r, p = stats.spearmanr(data[c1], data[c2])
                rows.append({
                    "var_1": c1,
                    "var_2": c2,
                    "correlation": r,
                    "p_value": p,
                    "significant": p < alpha,
                    "strength": self._strength(r),
                })
        return pd.DataFrame(rows).sort_values("p_value")

    def point_biserial(self, binary_col: str, continuous_col: str) -> dict:
        """Point-biserial correlation between a binary and continuous variable."""
        data = self.df[[binary_col, continuous_col]].dropna()
        r, p = stats.pointbiserialr(data[binary_col], data[continuous_col])
        return {"correlation": r, "p_value": p, "binary": binary_col, "continuous": continuous_col}

    def top_correlations(self, n: int = 10, method: str = "pearson") -> pd.DataFrame:
        """Return the top-n strongest correlations (absolute value)."""
        corr = self.correlation_matrix(method)
        # Upper triangle only
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        pairs = corr.where(mask).stack().reset_index()
        pairs.columns = ["var_1", "var_2", "correlation"]
        pairs["abs_corr"] = pairs["correlation"].abs()
        return pairs.nlargest(n, "abs_corr").drop(columns="abs_corr").reset_index(drop=True)

    @staticmethod
    def _strength(r: float) -> str:
        ar = abs(r)
        if ar >= 0.8:
            return "very_strong"
        if ar >= 0.6:
            return "strong"
        if ar >= 0.4:
            return "moderate"
        if ar >= 0.2:
            return "weak"
        return "negligible"
