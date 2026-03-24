"""Data cleansing pipeline utilities.

This module centralizes common preprocessing operations for tabular datasets:

- column-name normalization,
- string whitespace cleanup,
- missing-value profiling,
- missing-value imputation/removal with multiple strategies.

The main entry point is :class:`DataCleaner`, which follows a familiar
``fit/transform`` API similar to scikit-learn estimators.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from loguru import logger


class DataCleaner:
    """Configurable data-cleansing pipeline for :class:`pandas.DataFrame`.

    The cleaner can standardize column names, trim whitespace in string
    columns, and handle missing values using one of several strategies.
    For statistical imputation strategies (``mean``, ``median``, ``mode``),
    the learned fill values are computed in :meth:`fit` and reused in
    :meth:`transform`.

    Parameters
    ----------
    missing_strategy:
        Strategy used to handle missing values.

        - ``"drop"``: drop rows with any missing value.
        - ``"mean"``: fill numeric columns with column means.
        - ``"median"``: fill numeric columns with column medians.
        - ``"mode"``: fill columns with most frequent value.
        - ``"ffill"``: forward-fill followed by backward-fill.
        - ``"knn"``: KNN imputation on numeric columns.
    knn_neighbors:
        Number of neighbors used when ``missing_strategy="knn"``.
    standardize_columns:
        If ``True``, apply :meth:`_standardize_columns` before other steps.
    strip_whitespace:
        If ``True``, apply :meth:`_strip_whitespace` to object columns.
    """

    def __init__(
        self,
        missing_strategy: Literal["drop", "mean", "median", "mode", "ffill", "knn"] = "median",
        knn_neighbors: int = 5,
        standardize_columns: bool = True,
        strip_whitespace: bool = True,
    ) -> None:
        """Initialize cleansing configuration and internal state."""
        self.missing_strategy = missing_strategy
        self.knn_neighbors = knn_neighbors
        self.standardize_columns = standardize_columns
        self.strip_whitespace = strip_whitespace
        self._fill_values: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Learn imputation values from a training dataset.

        For ``mean``, ``median``, and ``mode`` strategies, this method stores
        per-column values in ``self._fill_values``. For the other strategies,
        fitting is a no-op and the cleaner is returned unchanged.

        Parameters
        ----------
        df:
            Training dataframe used to estimate imputation statistics.

        Returns
        -------
        DataCleaner
            The same cleaner instance (to allow method chaining).
        """
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
        """Run the configured cleansing pipeline.

        Processing order:

        1. Optional column standardization.
        2. Optional string whitespace trimming.
        3. Missing-value handling via :meth:`_handle_missing`.

        Parameters
        ----------
        df:
            Input dataframe.

        Returns
        -------
        pandas.DataFrame
            A new dataframe with the cleansing operations applied.
        """
        df = df.copy()
        if self.standardize_columns:
            df = self._standardize_columns(df)
        if self.strip_whitespace:
            df = self._strip_whitespace(df)
        df = self._handle_missing(df)
        logger.info("Cleansing complete — shape {}", df.shape)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit cleaner statistics and transform data in one call.

        Parameters
        ----------
        df:
            Input dataframe used both for fitting and transformation.

        Returns
        -------
        pandas.DataFrame
            Cleansed dataframe.
        """
        return self.fit(df).transform(df)

    def profile_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate a per-column missing-value summary.

        Parameters
        ----------
        df:
            Input dataframe to profile.

        Returns
        -------
        pandas.DataFrame
            Summary table with ``missing_count`` and ``missing_pct`` for each
            column containing at least one missing value, sorted descending by
            missing percentage.
        """
        total = len(df)
        missing = df.isnull().sum()
        pct = (missing / total) * 100
        return (
            pd.DataFrame({"missing_count": missing, "missing_pct": pct})
            .sort_values("missing_pct", ascending=False)
            .query("missing_count > 0")
        )

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to snake_case-like identifiers.

        The method strips surrounding whitespace, lowercases text, replaces
        non-word character groups with underscores, and trims underscores
        at the edges.
        """
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(r"[^\w]+", "_", regex=True)
            .str.strip("_")
        )
        return df

    @staticmethod
    def _strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
        """Trim leading/trailing whitespace in object/string columns."""
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip()
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply missing-value strategy configured on the cleaner.

        Notes
        -----
        For ``mean/median/mode`` strategies, numeric columns are imputed with
        fitted values when available, and categorical columns are filled with
        their current mode during transformation.
        """
        if self.missing_strategy == "drop":
            return df.dropna()
        if self.missing_strategy == "ffill":
            return df.ffill().bfill()
        if self.missing_strategy == "knn":
            return self._knn_impute(df)
        numeric = df.select_dtypes(include="number").columns
        for col in numeric:
            if col in self._fill_values:
                df[col] = df[col].fillna(self._fill_values[col])
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
        return df

    def _knn_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute numeric missing values with k-nearest neighbors.

        Non-numeric columns are preserved and concatenated back after numeric
        imputation.
        """
        from sklearn.impute import KNNImputer

        numeric = df.select_dtypes(include="number")
        non_numeric = df.select_dtypes(exclude="number")
        imputer = KNNImputer(n_neighbors=self.knn_neighbors)
        imputed = pd.DataFrame(
            imputer.fit_transform(numeric), columns=numeric.columns, index=numeric.index
        )
        return pd.concat([imputed, non_numeric], axis=1)
