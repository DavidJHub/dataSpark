"""Reusable visualization factory for EDA workflows.

The module defines :class:`PlotFactory` with static helper methods that build
common diagnostic plots and return Matplotlib figure objects for composition,
testing, and explicit save control.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


_DEFAULT_FIGSIZE = (10, 6)


class PlotFactory:
    """Generate standard EDA plots using Matplotlib + Seaborn.

    Parameters
    ----------
    style:
        Seaborn style name applied globally.
    figsize:
        Default figure size used by instance-level workflows.
    """

    def __init__(self, style: str = "whitegrid", figsize: tuple = (10, 6)) -> None:
        """Configure plotting style and store default figure size."""
        sns.set_style(style)
        self.figsize = figsize

    @staticmethod
    def missing_heatmap(df: pd.DataFrame, figsize: tuple = _DEFAULT_FIGSIZE) -> plt.Figure:
        """Plot missing-value matrix as heatmap.

        Parameters
        ----------
        df:
            Input dataframe.
        figsize:
            Figure dimensions.
        """
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap="viridis", ax=ax)
        ax.set_title("Missing Values Heatmap")
        plt.tight_layout()
        return fig

    @staticmethod
    def correlation_heatmap(
        df: pd.DataFrame, method: str = "pearson", figsize: tuple = _DEFAULT_FIGSIZE
    ) -> plt.Figure:
        """Plot upper-triangle correlation heatmap for numeric columns.

        Parameters
        ----------
        df:
            Input dataframe.
        method:
            Correlation method passed to ``DataFrame.corr``.
        figsize:
            Figure dimensions.
        """
        corr = df.select_dtypes(include="number").corr(method=method)
        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            ax=ax,
            square=True,
        )
        ax.set_title(f"{method.title()} Correlation Matrix")
        plt.tight_layout()
        return fig

    @staticmethod
    def distribution_grid(
        df: pd.DataFrame, columns: list[str] | None = None, figsize: tuple | None = None
    ) -> plt.Figure:
        """Render histogram + KDE panels for numeric columns.

        Parameters
        ----------
        df:
            Input dataframe.
        columns:
            Optional numeric column subset.
        figsize:
            Explicit figure size. If omitted, inferred from grid geometry.
        """
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
        """Render grid of boxplots for outlier inspection.

        Parameters
        ----------
        df:
            Input dataframe.
        columns:
            Optional numeric column subset.
        figsize:
            Explicit figure size. If omitted, inferred from grid geometry.
        """
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

    box_grid = boxplot_grid

    @staticmethod
    def pairplot(df: pd.DataFrame, hue: str | None = None) -> plt.Figure:
        """Create Seaborn pairplot for numeric variables.

        Parameters
        ----------
        df:
            Input dataframe.
        hue:
            Optional categorical column used for color grouping.
        """
        grid = sns.pairplot(df.select_dtypes(include="number"), hue=hue, diag_kind="kde")
        return grid.figure

    @staticmethod
    def save(fig: plt.Figure, path: str, dpi: int = 150) -> None:
        """Persist figure to disk and close it to release memory."""
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
