"""
Feature Engineering
===================
Custom sklearn transformers for feature creation,
selection, and dimensionality reduction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_regression
from loguru import logger


class FeatureEngineer:
    """Utility class for common feature engineering tasks."""

    @staticmethod
    def create_interaction_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Create pairwise multiplication interaction terms."""
        df = df.copy()
        for i, c1 in enumerate(columns):
            for c2 in columns[i + 1:]:
                df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
        return df

    @staticmethod
    def create_polynomial_features(
        df: pd.DataFrame, columns: list[str], degree: int = 2
    ) -> pd.DataFrame:
        """Add polynomial terms up to given degree."""
        df = df.copy()
        for col in columns:
            for d in range(2, degree + 1):
                df[f"{col}_pow{d}"] = df[col] ** d
        return df

    @staticmethod
    def create_log_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Log-transform (log1p) for skewed features."""
        df = df.copy()
        for col in columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
        return df

    @staticmethod
    def select_k_best(
        X: pd.DataFrame, y: pd.Series, k: int = 10, task: str = "classification"
    ) -> pd.DataFrame:
        """Select top-k features by mutual information or F-regression."""
        score_func = mutual_info_classif if task == "classification" else f_regression
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        selector.fit(X.select_dtypes(include="number"), y)
        scores = pd.DataFrame({
            "feature": X.select_dtypes(include="number").columns,
            "score": selector.scores_,
        }).sort_values("score", ascending=False)
        logger.info("Top {} features selected", k)
        return scores

    @staticmethod
    def pca_reduce(
        X: pd.DataFrame, n_components: int | float = 0.95
    ) -> tuple[np.ndarray, PCA]:
        """PCA dimensionality reduction."""
        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(X.select_dtypes(include="number"))
        logger.info(
            "PCA: {} → {} components (explained variance: {:.1f}%)",
            X.shape[1], pca.n_components_, pca.explained_variance_ratio_.sum() * 100,
        )
        return transformed, pca


class BinningTransformer(BaseEstimator, TransformerMixin):
    """Custom sklearn transformer: equal-frequency binning."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self._bin_edges: dict[int, np.ndarray] = {}

    def fit(self, X, y=None):
        X = np.asarray(X)
        for col_idx in range(X.shape[1]):
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            self._bin_edges[col_idx] = np.percentile(X[:, col_idx], quantiles)
        return self

    def transform(self, X):
        X = np.asarray(X)
        result = np.zeros_like(X)
        for col_idx in range(X.shape[1]):
            result[:, col_idx] = np.digitize(X[:, col_idx], self._bin_edges[col_idx]) - 1
        return result
