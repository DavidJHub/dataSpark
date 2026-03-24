"""
Visualization Factory
=====================
Reusable plotting functions built on Matplotlib + Seaborn.
All functions return Figure objects for composability.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI/server

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


_DEFAULT_FIGSIZE = (10, 6)


class PlotFactory:
    """Generate standard EDA plots."""

    def __init__(self, style: str = "whitegrid", figsize: tuple = (10, 6)) -> None:
        sns.set_style(style)
        self.figsize = figsize

    @staticmethod
    def missing_heatmap(df: pd.DataFrame, figsize: tuple = _DEFAULT_FIGSIZE) -> plt.Figure:
        """Heatmap of missing values."""
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap="viridis", ax=ax)
        ax.set_title("Missing Values Heatmap")
        plt.tight_layout()
        return fig

    @staticmethod
    def correlation_heatmap(
        df: pd.DataFrame, method: str = "pearson", figsize: tuple = _DEFAULT_FIGSIZE
    ) -> plt.Figure:
        """Correlation heatmap for numeric columns."""
        corr = df.select_dtypes(include="number").corr(method=method)
        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                     center=0, ax=ax, square=True)
        ax.set_title(f"{method.title()} Correlation Matrix")
        plt.tight_layout()
        return fig

    @staticmethod
    def distribution_grid(
        df: pd.DataFrame, columns: list[str] | None = None, figsize: tuple | None = None
    ) -> plt.Figure:
        """Grid of histograms with KDE for numeric columns."""
        cols = columns or df.select_dtypes(include="number").columns.tolist()
        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig_size = figsize or (5 * ncols, 4 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=fig_size)
        axes = np.atleast_1d(axes).flatten()
        for i, col in enumerate(cols):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        return fig

    @staticmethod
    def boxplot_grid(
        df: pd.DataFrame, columns: list[str] | None = None, figsize: tuple | None = None
    ) -> plt.Figure:
        """Grid of box plots for outlier visualization."""
        cols = columns or df.select_dtypes(include="number").columns.tolist()
        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig_size = figsize or (5 * ncols, 4 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=fig_size)
        axes = np.atleast_1d(axes).flatten()
        for i, col in enumerate(cols):
            sns.boxplot(y=df[col].dropna(), ax=axes[i])
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        return fig

    # Alias for notebook compatibility
    box_grid = boxplot_grid

    @staticmethod
    def pairplot(df: pd.DataFrame, hue: str | None = None) -> plt.Figure:
        """Seaborn pairplot for numeric columns."""
        grid = sns.pairplot(df.select_dtypes(include="number"), hue=hue, diag_kind="kde")
        return grid.figure

    @staticmethod
    def save(fig: plt.Figure, path: str, dpi: int = 150) -> None:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
