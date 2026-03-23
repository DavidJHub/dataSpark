"""
Data Explorer
=============
Comprehensive descriptive statistics and data profiling.
Uses Pandas + NumPy + SciPy for robust statistical summaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


class DataExplorer:
    """Generate comprehensive statistical profiles for DataFrames."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        logger.info("DataExplorer initialized — {} rows × {} cols", *df.shape)

    def summary(self) -> pd.DataFrame:
        """Extended describe() with skewness, kurtosis, and missing info."""
        numeric = self.df.select_dtypes(include="number")
        desc = numeric.describe().T
        desc["skewness"] = numeric.skew()
        desc["kurtosis"] = numeric.kurtosis()
        desc["missing_count"] = self.df[numeric.columns].isnull().sum()
        desc["missing_pct"] = (desc["missing_count"] / len(self.df)) * 100
        desc["iqr"] = desc["75%"] - desc["25%"]
        desc["cv"] = (desc["std"] / desc["mean"]).abs()  # coefficient of variation
        return desc

    def categorical_summary(self) -> dict[str, pd.DataFrame]:
        """Frequency tables for categorical columns."""
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns
        result = {}
        for col in cat_cols:
            freq = self.df[col].value_counts()
            pct = self.df[col].value_counts(normalize=True) * 100
            result[col] = pd.DataFrame({"count": freq, "pct": pct})
        return result

    def normality_tests(self, alpha: float = 0.05) -> pd.DataFrame:
        """Shapiro-Wilk normality test for each numeric column."""
        rows = []
        for col in self.df.select_dtypes(include="number").columns:
            data = self.df[col].dropna()
            if len(data) < 8 or len(data) > 5000:
                # Shapiro-Wilk limited to ~5000 samples; use D'Agostino for larger
                if len(data) >= 20:
                    stat, p = stats.normaltest(data)
                    test_name = "DAgostino-Pearson"
                else:
                    continue
            else:
                stat, p = stats.shapiro(data)
                test_name = "Shapiro-Wilk"
            rows.append({
                "column": col,
                "test": test_name,
                "statistic": stat,
                "p_value": p,
                "is_normal": p > alpha,
            })
        return pd.DataFrame(rows)

    def info_report(self) -> dict:
        """High-level dataset info dictionary."""
        return {
            "shape": self.df.shape,
            "memory_mb": self.df.memory_usage(deep=True).sum() / 1e6,
            "dtypes": self.df.dtypes.value_counts().to_dict(),
            "total_missing": int(self.df.isnull().sum().sum()),
            "total_missing_pct": self.df.isnull().mean().mean() * 100,
            "duplicate_rows": int(self.df.duplicated().sum()),
            "numeric_columns": self.df.select_dtypes(include="number").columns.tolist(),
            "categorical_columns": self.df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist(),
        }
