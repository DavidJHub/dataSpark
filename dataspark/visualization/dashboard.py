"""
Dashboard
=========
Compose multi-panel summary dashboards from ChartBuilder plots
for quick data overviews and reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from dataspark.visualization.themes import Theme


class Dashboard:
    """Create multi-panel summary dashboards."""

    def __init__(self, theme: Theme | None = None) -> None:
        self.theme = theme or Theme()
        self.theme.apply()

    def data_quality(self, df: pd.DataFrame) -> plt.Figure:
        """4-panel data-quality overview: missing matrix, missing bars,
        dtype pie chart, and duplicate summary."""
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        # Panel 1: missing matrix
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(df.isnull().values, aspect="auto", cmap="gray_r", interpolation="none")
        ax1.set_yticks([])
        ax1.set_xticks(range(len(df.columns)))
        ax1.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
        ax1.set_title("Missing Pattern")

        # Panel 2: missing % bars
        ax2 = fig.add_subplot(gs[0, 1])
        pct = (df.isnull().mean() * 100).sort_values(ascending=True)
        colors = [
            self.theme.danger if v > 30
            else self.theme.warning if v > 10
            else self.theme.primary
            for v in pct.values
        ]
        ax2.barh(pct.index, pct.values, color=colors)
        ax2.set_xlabel("Missing %")
        ax2.set_title("Missing by Column")

        # Panel 3: dtype breakdown
        ax3 = fig.add_subplot(gs[1, 0])
        dtype_counts = df.dtypes.astype(str).value_counts()
        ax3.pie(dtype_counts.values, labels=dtype_counts.index, autopct="%1.0f%%",
                colors=self.theme.categorical_palette[:len(dtype_counts)])
        ax3.set_title("Data Type Breakdown")

        # Panel 4: key metrics text
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")
        metrics = [
            f"Rows: {len(df):,}",
            f"Columns: {len(df.columns)}",
            f"Numeric: {len(df.select_dtypes(include='number').columns)}",
            f"Categorical: {len(df.select_dtypes(include=['object', 'category']).columns)}",
            f"Total Missing: {df.isnull().sum().sum():,} ({df.isnull().mean().mean()*100:.1f}%)",
            f"Duplicate Rows: {df.duplicated().sum():,}",
            f"Memory: {df.memory_usage(deep=True).sum()/1e6:.1f} MB",
        ]
        for i, line in enumerate(metrics):
            ax4.text(0.1, 0.9 - i * 0.12, line, fontsize=12,
                     transform=ax4.transAxes, family="monospace")
        ax4.set_title("Dataset Summary")

        fig.suptitle("Data Quality Dashboard", fontsize=16, fontweight="bold")
        return fig

    def eda_overview(self, df: pd.DataFrame) -> plt.Figure:
        """6-panel EDA overview: distributions, correlation heatmap,
        categorical bars, box plots, and basic stats."""
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        n_num = min(len(numeric_cols), 4)

        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

        # Panel 1: distribution grid (top-4 numeric)
        ax1 = fig.add_subplot(gs[0, 0])
        for i, col in enumerate(numeric_cols[:4]):
            color = self.theme.categorical_palette[i]
            sns.kdeplot(df[col].dropna(), ax=ax1, label=col, color=color, fill=True, alpha=0.25)
        ax1.set_title("Numeric Distributions (KDE)")
        ax1.legend(fontsize=8)

        # Panel 2: correlation heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=len(numeric_cols) <= 8, fmt=".2f",
                        cmap=self.theme.diverging_cmap, center=0, ax=ax2,
                        square=True, linewidths=0.5, cbar_kws={"shrink": 0.7})
        ax2.set_title("Correlation Matrix")

        # Panel 3: box plots
        ax3 = fig.add_subplot(gs[1, 0])
        if numeric_cols:
            # Z-score normalize for comparable box plots
            normed = (df[numeric_cols[:6]] - df[numeric_cols[:6]].mean()) / df[numeric_cols[:6]].std()
            sns.boxplot(data=normed, ax=ax3, palette=self.theme.categorical_palette)
            ax3.set_title("Box Plots (Z-normalized)")
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

        # Panel 4: top categorical
        ax4 = fig.add_subplot(gs[1, 1])
        if cat_cols:
            top_cat = cat_cols[0]
            counts = df[top_cat].value_counts().head(10)
            counts.plot.bar(ax=ax4, color=self.theme.primary, alpha=0.75)
            ax4.set_title(f"Top Categories — {top_cat}")
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right")
        else:
            ax4.text(0.5, 0.5, "No categorical columns", ha="center", va="center",
                     transform=ax4.transAxes)
            ax4.set_axis_off()

        # Panel 5: skewness / kurtosis bars
        ax5 = fig.add_subplot(gs[2, 0])
        if numeric_cols:
            skew = df[numeric_cols].skew().sort_values()
            colors = [self.theme.danger if abs(v) > 1 else self.theme.primary for v in skew.values]
            skew.plot.barh(ax=ax5, color=colors)
            ax5.axvline(0, color="black", lw=0.5)
            ax5.set_title("Skewness (|>1| highlighted)")

        # Panel 6: pairwise scatter of top 2 correlated
        ax6 = fig.add_subplot(gs[2, 1])
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().abs()
            corr_arr = corr.to_numpy().copy()
            np.fill_diagonal(corr_arr, 0)
            idx = np.unravel_index(corr_arr.argmax(), corr_arr.shape)
            c1, c2 = numeric_cols[idx[0]], numeric_cols[idx[1]]
            ax6.scatter(df[c1], df[c2], s=10, alpha=0.4, color=self.theme.primary)
            ax6.set_xlabel(c1)
            ax6.set_ylabel(c2)
            r, _ = stats.pearsonr(df[c1].dropna(), df[c2].dropna())
            ax6.set_title(f"Top Pair: {c1} vs {c2} (r={r:.2f})")

        fig.suptitle("Exploratory Data Analysis", fontsize=16, fontweight="bold")
        return fig

    def model_report(
        self,
        comparison_df: pd.DataFrame,
        feature_scores: pd.DataFrame | None = None,
        y_true: np.ndarray | None = None,
        y_pred: np.ndarray | None = None,
    ) -> plt.Figure:
        """ML model evaluation dashboard: model comparison,
        feature importance, confusion or residual analysis."""
        n_panels = 2 + (1 if feature_scores is not None else 0) + (1 if y_true is not None else 0)
        n_cols = 2
        n_rows = (n_panels + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
        axes = axes.flatten()
        idx = 0

        # Panel: model comparison bars
        ax = axes[idx]; idx += 1
        x = np.arange(len(comparison_df))
        w = 0.35
        ax.bar(x - w / 2, comparison_df["train_score_mean"], w, label="Train",
               color=self.theme.primary, alpha=0.7)
        ax.bar(x + w / 2, comparison_df["test_score_mean"], w, label="Test",
               color=self.theme.secondary, alpha=0.7,
               yerr=comparison_df.get("test_score_std", 0), capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df["model"], rotation=25, ha="right")
        ax.set_title("Model Comparison")
        ax.legend()

        # Panel: overfitting gap
        ax = axes[idx]; idx += 1
        ax.bar(comparison_df["model"], comparison_df["overfit_gap"],
               color=[self.theme.danger if g > 0.1 else self.theme.success
                      for g in comparison_df["overfit_gap"]])
        ax.set_title("Overfit Gap (Train - Test)")
        ax.axhline(0, color="black", lw=0.5)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")

        # Panel: feature importance
        if feature_scores is not None:
            ax = axes[idx]; idx += 1
            top = feature_scores.nlargest(10, "score")
            ax.barh(top["feature"], top["score"], color=self.theme.primary)
            ax.set_title("Top Features")
            ax.invert_yaxis()

        # Panel: confusion or residuals
        if y_true is not None and y_pred is not None:
            ax = axes[idx]; idx += 1
            unique = np.unique(y_true)
            if len(unique) <= 20:
                from sklearn.metrics import confusion_matrix as cm_fn
                cm = cm_fn(y_true, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap=self.theme.sequential_cmap, ax=ax)
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            else:
                resid = y_true - y_pred
                ax.scatter(y_pred, resid, s=10, alpha=0.4, color=self.theme.primary)
                ax.axhline(0, color=self.theme.danger, lw=1)
                ax.set_title("Residuals vs Predicted")

        # Hide unused axes
        for j in range(idx, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Model Evaluation Report", fontsize=16, fontweight="bold")
        fig.tight_layout()
        return fig

    def timeseries_report(
        self,
        series: pd.Series,
        components: dict | None = None,
        forecast: pd.Series | None = None,
    ) -> plt.Figure:
        """Time-series summary: original + rolling, decomposition, ACF."""
        n_rows = 1 + (1 if components else 0) + (1 if forecast is not None else 0)
        fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4 * n_rows), sharex=False)
        if n_rows == 1:
            axes = [axes]
        idx = 0

        # Panel 1: original + rolling
        ax = axes[idx]; idx += 1
        ax.plot(series.index, series.values, color=self.theme.primary, alpha=0.5, lw=0.8)
        window = max(7, len(series) // 20)
        rm = series.rolling(window).mean()
        ax.plot(rm.index, rm.values, color=self.theme.danger, lw=2,
                label=f"Rolling Mean ({window})")
        ax.legend()
        ax.set_title(series.name or "Time Series")

        # Panel 2: decomposition
        if components:
            ax = axes[idx]; idx += 1
            for key, color in [("trend", self.theme.danger), ("seasonal", self.theme.success)]:
                data = components.get(key)
                if data is not None:
                    ax.plot(data.index, data.values, label=key.title(), color=color, lw=1)
            ax.legend()
            ax.set_title("Trend + Seasonal Components")

        # Panel 3: forecast
        if forecast is not None:
            ax = axes[idx]; idx += 1
            ax.plot(series.index, series.values, color=self.theme.primary,
                    alpha=0.5, lw=0.8, label="Observed")
            ax.plot(forecast.index, forecast.values, color=self.theme.danger,
                    ls="--", lw=2, label="Forecast")
            ax.legend()
            ax.set_title("Forecast")

        fig.suptitle("Time Series Report", fontsize=16, fontweight="bold")
        fig.tight_layout()
        return fig
