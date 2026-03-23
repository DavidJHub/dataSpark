"""
Data Cleansing Pipeline
=======================
Handles missing values, type coercion, whitespace normalization,
and column standardization for tabular data.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger


class DataCleaner:
    """Configurable data-cleansing pipeline for pandas DataFrames."""

    def __init__(
        self,
        missing_strategy: Literal["drop", "mean", "median", "mode", "ffill", "knn"] = "median",
        knn_neighbors: int = 5,
        standardize_columns: bool = True,
        strip_whitespace: bool = True,
    ) -> None:
        self.missing_strategy = missing_strategy
        self.knn_neighbors = knn_neighbors
        self.standardize_columns = standardize_columns
        self.strip_whitespace = strip_whitespace
        self._fill_values: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Learn fill values from training data (for mean/median/mode)."""
        numeric = df.select_dtypes(include="number")
        if self.missing_strategy == "mean":
            self._fill_values = numeric.mean().to_dict()
        elif self.missing_strategy == "median":
            self._fill_values = numeric.median().to_dict()
        elif self.missing_strategy == "mode":
            self._fill_values = {c: df[c].mode().iloc[0] for c in df.columns if df[c].mode().size}
        logger.info("DataCleaner fitted with strategy={}", self.missing_strategy)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full cleansing pipeline and return a clean DataFrame."""
        df = df.copy()
        if self.standardize_columns:
            df = self._standardize_columns(df)
        if self.strip_whitespace:
            df = self._strip_whitespace(df)
        df = self._handle_missing(df)
        logger.info("Cleansing complete — shape {}", df.shape)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def profile_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a summary of missing values per column."""
        total = len(df)
        missing = df.isnull().sum()
        pct = (missing / total) * 100
        return (
            pd.DataFrame({"missing_count": missing, "missing_pct": pct})
            .sort_values("missing_pct", ascending=False)
            .query("missing_count > 0")
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(r"[^\w]+", "_", regex=True)
            .str.strip("_")
        )
        return df

    @staticmethod
    def _strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip()
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.missing_strategy == "drop":
            return df.dropna()
        if self.missing_strategy == "ffill":
            return df.ffill().bfill()
        if self.missing_strategy == "knn":
            return self._knn_impute(df)
        # mean / median / mode
        numeric = df.select_dtypes(include="number").columns
        for col in numeric:
            if col in self._fill_values:
                df[col] = df[col].fillna(self._fill_values[col])
        # categorical → mode
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
        return df

    def _knn_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        from sklearn.impute import KNNImputer

        numeric = df.select_dtypes(include="number")
        non_numeric = df.select_dtypes(exclude="number")
        imputer = KNNImputer(n_neighbors=self.knn_neighbors)
        imputed = pd.DataFrame(
            imputer.fit_transform(numeric), columns=numeric.columns, index=numeric.index
        )
        return pd.concat([imputed, non_numeric], axis=1)
