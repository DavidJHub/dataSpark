"""
Outlier Detection
=================
Multiple strategies: IQR, Z-score, Modified Z-score (MAD), Isolation Forest.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger


class OutlierDetector:
    """Detect and optionally remove outliers from numeric columns."""

    def __init__(
        self,
        method: Literal["iqr", "zscore", "mad", "isolation_forest"] = "iqr",
        threshold: float = 1.5,
        contamination: float = 0.05,
        *,
        factor: float | None = None,
    ) -> None:
        self.method = method
        self.threshold = factor if factor is not None else threshold
        self.contamination = contamination

    def detect(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        """Return boolean mask — True where value is an outlier."""
        cols = columns or df.select_dtypes(include="number").columns.tolist()
        mask = pd.DataFrame(False, index=df.index, columns=cols)
        for col in cols:
            series = df[col].dropna()
            if self.method == "iqr":
                mask[col] = self._iqr(df[col])
            elif self.method == "zscore":
                mask[col] = self._zscore(df[col])
            elif self.method == "mad":
                mask[col] = self._mad(df[col])
            elif self.method == "isolation_forest":
                mask[col] = self._iforest(df[col])
        n_outliers = mask.sum().sum()
        logger.info("Detected {} outliers via {} method", n_outliers, self.method)
        return mask

    def remove(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        """Remove rows containing any outlier in the specified columns."""
        mask = self.detect(df, columns)
        keep = ~mask.any(axis=1)
        removed = (~keep).sum()
        logger.info("Removed {} rows with outliers", removed)
        return df.loc[keep].reset_index(drop=True)

    def cap(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        """Winsorize: cap outliers at the boundary values instead of removing."""
        df = df.copy()
        cols = columns or df.select_dtypes(include="number").columns.tolist()
        for col in cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - self.threshold * iqr, q3 + self.threshold * iqr
            df[col] = df[col].clip(lower, upper)
        return df

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _iqr(self, s: pd.Series) -> pd.Series:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return (s < q1 - self.threshold * iqr) | (s > q3 + self.threshold * iqr)

    def _zscore(self, s: pd.Series) -> pd.Series:
        z = (s - s.mean()) / s.std()
        return z.abs() > self.threshold

    def _mad(self, s: pd.Series) -> pd.Series:
        median = s.median()
        mad = np.median(np.abs(s - median))
        modified_z = 0.6745 * (s - median) / (mad + 1e-10)
        return modified_z.abs() > self.threshold

    def _iforest(self, s: pd.Series) -> pd.Series:
        from sklearn.ensemble import IsolationForest

        clf = IsolationForest(contamination=self.contamination, random_state=42)
        vals = s.dropna().values.reshape(-1, 1)
        labels = pd.Series(1, index=s.index)
        labels.loc[s.dropna().index] = clf.fit_predict(vals)
        return labels == -1
