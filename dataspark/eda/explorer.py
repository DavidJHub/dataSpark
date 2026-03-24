"""Exploratory data profiling utilities.

This module defines :class:`DataExplorer`, a compact helper for descriptive
statistics, categorical frequency summaries, normality testing, and high-level
dataset diagnostics.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger
from scipy import stats


class DataExplorer:
    """Generate comprehensive statistical profiles for a dataframe.

    Parameters
    ----------
    df:
        Input dataframe to analyze.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Store dataframe reference and log its shape."""
        self.df = df
        logger.info("DataExplorer initialized — {} rows × {} cols", *df.shape)

    def summary(self) -> pd.DataFrame:
        """Return extended descriptive statistics for numeric columns.

        Includes standard ``describe()`` metrics plus skewness, kurtosis,
        missing count/percentage, interquartile range (IQR), and coefficient
        of variation (CV).
        """
        numeric = self.df.select_dtypes(include="number")
        desc = numeric.describe().T
        desc["skewness"] = numeric.skew()
        desc["kurtosis"] = numeric.kurtosis()
        desc["missing_count"] = self.df[numeric.columns].isnull().sum()
        desc["missing_pct"] = (desc["missing_count"] / len(self.df)) * 100
        desc["iqr"] = desc["75%"] - desc["25%"]
        desc["cv"] = (desc["std"] / desc["mean"]).abs()
        return desc

    def categorical_summary(self) -> dict[str, pd.DataFrame]:
        """Return frequency and percentage tables for categorical columns.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Dictionary keyed by column name, each value containing ``count``
            and ``pct`` for observed categories.
        """
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns
        result: dict[str, pd.DataFrame] = {}
        for col in cat_cols:
            freq = self.df[col].value_counts()
            pct = self.df[col].value_counts(normalize=True) * 100
            result[col] = pd.DataFrame({"count": freq, "pct": pct})
        return result

    def normality_tests(self, alpha: float = 0.05) -> pd.DataFrame:
        """Run per-column normality tests on numeric variables.

        Strategy:

        - For sample sizes in [8, 5000], uses Shapiro-Wilk.
        - For larger samples (>=20), uses D'Agostino-Pearson test.
        - Very small samples are skipped.

        Parameters
        ----------
        alpha:
            Significance threshold used to flag ``is_normal``.

        Returns
        -------
        pandas.DataFrame
            Table with test used, statistic, p-value, and normality flag.
        """
        rows = []
        for col in self.df.select_dtypes(include="number").columns:
            data = self.df[col].dropna()
            if len(data) < 8 or len(data) > 5000:
                if len(data) >= 20:
                    stat, p = stats.normaltest(data)
                    test_name = "DAgostino-Pearson"
                else:
                    continue
            else:
                stat, p = stats.shapiro(data)
                test_name = "Shapiro-Wilk"
            rows.append(
                {
                    "column": col,
                    "test": test_name,
                    "statistic": stat,
                    "p_value": p,
                    "is_normal": p > alpha,
                }
            )
        return pd.DataFrame(rows)

    def info_report(self) -> dict:
        """Build a high-level metadata report for the dataset.

        Returns
        -------
        dict
            Summary with shape, memory footprint, dtype distribution, missing
            values, duplicate rows, and lists of numeric/categorical columns.
        """
        return {
            "shape": self.df.shape,
            "memory_mb": self.df.memory_usage(deep=True).sum() / 1e6,
            "dtypes": self.df.dtypes.value_counts().to_dict(),
            "total_missing": int(self.df.isnull().sum().sum()),
            "total_missing_pct": self.df.isnull().mean().mean() * 100,
            "duplicate_rows": int(self.df.duplicated().sum()),
            "numeric_columns": self.df.select_dtypes(include="number").columns.tolist(),
            "categorical_columns": self.df.select_dtypes(include=["object", "category"]).columns.tolist(),
        }
