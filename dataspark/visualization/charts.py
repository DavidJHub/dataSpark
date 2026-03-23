"""
Chart Builder
=============
Domain-aware static chart builder that produces Matplotlib figures
from DataSpark module outputs.  Every method returns a plt.Figure
so callers can display, save, or compose into dashboards.

Domains covered:
  - Data profiling & cleansing  (missing values, outliers, type report)
  - EDA                         (distributions, correlations, categorical)
  - Statistical testing         (p-value forest, effect-size bars, QQ-plot)
  - ML pipelines                (model comparison, feature importance, learning curve)
  - Time series                 (decomposition, forecast, ACF/PACF)
  - Sampling                    (bootstrap CI, sample-size curve, strata proportions)
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats

from dataspark.visualization.themes import Theme


class ChartBuilder:
    """Produce publication-ready charts from DataSpark data structures."""

    def __init__(self, theme: Theme | None = None) -> None:
        self.theme = theme or Theme()
        self.theme.apply()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _fig(self, size: str = "default") -> tuple[plt.Figure, plt.Axes]:
        sizes = {
            "default": self.theme.figsize,
            "wide": self.theme.figsize_wide,
            "tall": self.theme.figsize_tall,
        }
        return plt.subplots(figsize=sizes.get(size, self.theme.figsize))

    @staticmethod
    def _close(fig: plt.Figure) -> plt.Figure:
        fig.tight_layout()
        return fig

    # ==================================================================
    # DATA PROFILING & CLEANSING
    # ==================================================================

    def missing_matrix(self, df: pd.DataFrame) -> plt.Figure:
        """Nullity matrix — white cells = missing."""
        fig, ax = self._fig("wide")
        ax.imshow(df.isnull().values, aspect="auto", cmap="gray_r", interpolation="none")
        ax.set_yticks([])
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right")
        ax.set_title("Missing-Value Matrix")
        return self._close(fig)

    def missing_bar(self, df: pd.DataFrame) -> plt.Figure:
        """Horizontal bar chart of missing-value percentages per column."""
        pct = (df.isnull().mean() * 100).sort_values()
        pct = pct[pct > 0]
        if pct.empty:
            fig, ax = self._fig()
            ax.text(0.5, 0.5, "No missing values", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.set_axis_off()
            return self._close(fig)
        colors = [self.theme.danger if v > 30 else self.theme.warning if v > 10
                  else self.theme.primary for v in pct.values]
        fig, ax = self._fig()
        pct.plot.barh(ax=ax, color=colors)
        ax.set_xlabel("Missing %")
        ax.set_title("Missing Values by Column")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        return self._close(fig)

    def outlier_scatter(
        self,
        df: pd.DataFrame,
        column: str,
        outlier_mask: pd.Series,
    ) -> plt.Figure:
        """Scatter plot highlighting outlier points."""
        fig, ax = self._fig("wide")
        normal = ~outlier_mask
        ax.scatter(df.index[normal], df.loc[normal, column],
                   c=self.theme.primary, alpha=0.5, s=15, label="Normal")
        ax.scatter(df.index[outlier_mask], df.loc[outlier_mask, column],
                   c=self.theme.danger, alpha=0.8, s=30, label="Outlier",
                   edgecolors="black", linewidths=0.5)
        ax.set_ylabel(column)
        ax.set_xlabel("Index")
        ax.set_title(f"Outlier Detection — {column}")
        ax.legend()
        return self._close(fig)

    def before_after_cleaning(
        self,
        before: pd.DataFrame,
        after: pd.DataFrame,
        column: str,
    ) -> plt.Figure:
        """Side-by-side distribution of a column before and after cleansing."""
        fig, axes = plt.subplots(1, 2, figsize=self.theme.figsize_wide, sharey=True)
        for ax, data, label, color in [
            (axes[0], before, "Before", self.theme.neutral),
            (axes[1], after, "After", self.theme.success),
        ]:
            sns.histplot(data[column].dropna(), kde=True, ax=ax, color=color, alpha=0.7)
            ax.set_title(f"{label} Cleaning")
            ax.set_xlabel(column)
        fig.suptitle(f"Distribution of '{column}' — Before vs After", fontsize=self.theme.title_size)
        return self._close(fig)

    # ==================================================================
    # EDA — DISTRIBUTIONS & CORRELATIONS
    # ==================================================================

    def distribution(
        self,
        series: pd.Series,
        bins: int = 30,
        kde: bool = True,
        fit_dist: str | None = None,
    ) -> plt.Figure:
        """Histogram + KDE with optional fitted distribution overlay."""
        fig, ax = self._fig()
        data = series.dropna()
        sns.histplot(data, bins=bins, kde=kde, stat="density", ax=ax,
                     color=self.theme.primary, alpha=self.theme.alpha)
        if fit_dist:
            dist_obj = getattr(stats, fit_dist, None)
            if dist_obj:
                params = dist_obj.fit(data)
                x = np.linspace(data.min(), data.max(), 200)
                ax.plot(x, dist_obj.pdf(x, *params), color=self.theme.danger,
                        lw=2, label=f"Fitted {fit_dist}")
                ax.legend()
        ax.set_title(f"Distribution — {series.name or 'values'}")
        return self._close(fig)

    def correlation_heatmap(
        self,
        df: pd.DataFrame,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        annot: bool = True,
    ) -> plt.Figure:
        """Lower-triangle correlation heatmap."""
        corr = df.select_dtypes(include="number").corr(method=method)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=self.theme.figsize_tall)
        sns.heatmap(corr, mask=mask, annot=annot, fmt=".2f",
                    cmap=self.theme.diverging_cmap, center=0,
                    ax=ax, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8})
        ax.set_title(f"{method.title()} Correlation Matrix")
        return self._close(fig)

    def top_correlations_bar(
        self,
        pairs: pd.DataFrame,
        n: int = 15,
    ) -> plt.Figure:
        """Horizontal bar chart of strongest correlations.

        Parameters
        ----------
        pairs : DataFrame with columns ``var_1``, ``var_2``, ``correlation``
            (output of ``CorrelationAnalyzer.top_correlations``).
        """
        data = pairs.head(n).copy()
        data["label"] = data["var_1"] + " ↔ " + data["var_2"]
        colors = [self.theme.primary if v >= 0 else self.theme.danger
                  for v in data["correlation"]]
        fig, ax = self._fig()
        ax.barh(data["label"], data["correlation"], color=colors)
        ax.set_xlabel("Correlation")
        ax.set_title(f"Top-{n} Correlations")
        ax.axvline(0, color="black", lw=0.8)
        ax.invert_yaxis()
        return self._close(fig)

    def categorical_bars(
        self,
        df: pd.DataFrame,
        column: str,
        hue: str | None = None,
        top_n: int = 20,
    ) -> plt.Figure:
        """Value-count bar chart with optional hue grouping."""
        fig, ax = self._fig()
        if hue:
            ct = pd.crosstab(df[column], df[hue])
            ct = ct.loc[ct.sum(axis=1).nlargest(top_n).index]
            ct.plot.bar(stacked=True, ax=ax, color=self.theme.categorical_palette)
            ax.legend(title=hue)
        else:
            counts = df[column].value_counts().head(top_n)
            counts.plot.bar(ax=ax, color=self.theme.primary, alpha=self.theme.alpha)
        ax.set_title(f"Category Distribution — {column}")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        return self._close(fig)

    def qq_plot(self, series: pd.Series) -> plt.Figure:
        """Quantile-quantile plot against the normal distribution."""
        fig, ax = self._fig()
        data = series.dropna().values
        stats.probplot(data, dist="norm", plot=ax)
        ax.get_lines()[0].set(color=self.theme.primary, markersize=4, alpha=0.6)
        ax.get_lines()[1].set(color=self.theme.danger, linewidth=1.5)
        ax.set_title(f"Q-Q Plot — {series.name or 'values'}")
        return self._close(fig)

    # ==================================================================
    # STATISTICAL TESTING
    # ==================================================================

    def p_value_forest(self, test_results: list[dict]) -> plt.Figure:
        """Forest plot of p-values from multiple hypothesis tests.

        Parameters
        ----------
        test_results : list of dicts, each with keys ``test`` and ``p_value``.
        """
        names = [r["test"] for r in test_results]
        pvals = [r["p_value"] for r in test_results]
        colors = [self.theme.success if p >= 0.05 else self.theme.danger for p in pvals]

        fig, ax = self._fig()
        y = range(len(names))
        ax.barh(y, [-np.log10(p + 1e-300) for p in pvals], color=colors, height=0.6)
        ax.axvline(-np.log10(0.05), color="black", ls="--", lw=1, label="α = 0.05")
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("-log₁₀(p)")
        ax.set_title("Hypothesis Test Results")
        ax.legend()
        ax.invert_yaxis()
        return self._close(fig)

    def effect_size_bar(self, results: list[dict]) -> plt.Figure:
        """Bar chart of effect sizes (Cohen's d, Cramér's V, etc.).

        Parameters
        ----------
        results : list of dicts, each with ``label`` and ``value`` keys.
        """
        labels = [r["label"] for r in results]
        values = [r["value"] for r in results]
        fig, ax = self._fig()
        bars = ax.barh(labels, values, color=self.theme.secondary, height=0.5)
        # Threshold lines for Cohen's d conventions
        for thresh, lbl in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
            ax.axvline(thresh, color=self.theme.neutral, ls=":", lw=0.8, alpha=0.7)
            ax.text(thresh, len(labels) - 0.2, lbl, fontsize=8, color=self.theme.neutral, ha="center")
        ax.set_xlabel("Effect Size")
        ax.set_title("Effect Sizes")
        ax.invert_yaxis()
        return self._close(fig)

    def group_comparison(
        self,
        groups: dict[str, np.ndarray | pd.Series],
        kind: Literal["box", "violin", "strip"] = "violin",
    ) -> plt.Figure:
        """Compare distributions across named groups."""
        long = pd.DataFrame([
            {"group": name, "value": v}
            for name, arr in groups.items()
            for v in np.asarray(arr)
        ])
        fig, ax = self._fig()
        plot_fn = {"box": sns.boxplot, "violin": sns.violinplot, "strip": sns.stripplot}[kind]
        plot_fn(data=long, x="group", y="value", ax=ax,
                palette=self.theme.categorical_palette[:len(groups)])
        ax.set_title("Group Comparison")
        return self._close(fig)

    # ==================================================================
    # ML PIPELINES
    # ==================================================================

    def model_comparison(self, results: pd.DataFrame) -> plt.Figure:
        """Bar chart with error bars comparing model cross-validation scores.

        Parameters
        ----------
        results : DataFrame from ``ModelSelector.compare_models()``.
        """
        fig, ax = self._fig("wide")
        x = np.arange(len(results))
        width = 0.35
        ax.bar(x - width / 2, results["train_score_mean"], width, label="Train",
               color=self.theme.primary, alpha=0.7)
        ax.bar(x + width / 2, results["test_score_mean"], width, label="Test",
               color=self.theme.secondary, alpha=0.7,
               yerr=results.get("test_score_std", 0), capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(results["model"], rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison (CV)")
        ax.legend()
        return self._close(fig)

    def feature_importance(
        self,
        scores: pd.DataFrame,
        top_n: int = 15,
    ) -> plt.Figure:
        """Horizontal bar of feature importance scores.

        Parameters
        ----------
        scores : DataFrame from ``FeatureEngineer.select_k_best()``
                 with columns ``feature`` and ``score``.
        """
        data = scores.nlargest(top_n, "score")
        fig, ax = self._fig()
        ax.barh(data["feature"], data["score"], color=self.theme.primary)
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Top-{top_n} Feature Importances")
        ax.invert_yaxis()
        return self._close(fig)

    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[str] | None = None,
    ) -> plt.Figure:
        """Annotated confusion matrix heatmap."""
        from sklearn.metrics import confusion_matrix as cm_fn

        cm = cm_fn(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(max(6, cm.shape[0]), max(5, cm.shape[0])))
        sns.heatmap(cm, annot=True, fmt="d", cmap=self.theme.sequential_cmap,
                    xticklabels=labels or "auto",
                    yticklabels=labels or "auto", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        return self._close(fig)

    def pca_variance(self, explained_variance_ratio: np.ndarray) -> plt.Figure:
        """Scree plot + cumulative explained variance."""
        fig, ax = self._fig()
        n = len(explained_variance_ratio)
        x = range(1, n + 1)
        cumulative = np.cumsum(explained_variance_ratio)
        ax.bar(x, explained_variance_ratio, color=self.theme.primary, alpha=0.7, label="Individual")
        ax.plot(x, cumulative, "o-", color=self.theme.danger, label="Cumulative")
        ax.axhline(0.95, color=self.theme.neutral, ls="--", lw=0.8, label="95% threshold")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("PCA — Explained Variance")
        ax.legend()
        return self._close(fig)

    def residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> plt.Figure:
        """Residual analysis: residuals vs predicted + histogram."""
        resid = np.asarray(y_true) - np.asarray(y_pred)
        fig, axes = plt.subplots(1, 2, figsize=self.theme.figsize_wide)

        axes[0].scatter(y_pred, resid, alpha=0.4, s=15, color=self.theme.primary)
        axes[0].axhline(0, color=self.theme.danger, lw=1)
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuals vs Predicted")

        sns.histplot(resid, kde=True, ax=axes[1], color=self.theme.secondary)
        axes[1].set_title("Residual Distribution")
        return self._close(fig)

    # ==================================================================
    # TIME SERIES
    # ==================================================================

    def ts_line(
        self,
        series: pd.Series,
        title: str | None = None,
        rolling_window: int | None = None,
    ) -> plt.Figure:
        """Simple time-series line plot with optional rolling mean."""
        fig, ax = self._fig("wide")
        ax.plot(series.index, series.values, color=self.theme.primary,
                alpha=0.6, lw=0.8, label="Observed")
        if rolling_window:
            rm = series.rolling(rolling_window).mean()
            ax.plot(rm.index, rm.values, color=self.theme.danger,
                    lw=2, label=f"Rolling Mean ({rolling_window})")
            ax.legend()
        ax.set_title(title or (series.name or "Time Series"))
        ax.set_xlabel("Date")
        return self._close(fig)

    def ts_decomposition(self, components: dict) -> plt.Figure:
        """4-panel time-series decomposition plot.

        Parameters
        ----------
        components : dict from ``TimeSeriesDecomposer.decompose()``.
        """
        panels = ["observed", "trend", "seasonal", "residual"]
        colors = [self.theme.primary, self.theme.danger, self.theme.success, self.theme.neutral]
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        for ax, key, color in zip(axes, panels, colors):
            data = components[key]
            if data is not None:
                ax.plot(data.index, data.values, color=color, lw=1)
            ax.set_ylabel(key.title())
        fig.suptitle(f"Time Series Decomposition (period={components.get('period', '?')})",
                     fontsize=self.theme.title_size)
        return self._close(fig)

    def ts_forecast(
        self,
        observed: pd.Series,
        forecast: pd.Series,
        ci_lower: pd.Series | None = None,
        ci_upper: pd.Series | None = None,
    ) -> plt.Figure:
        """Observed + forecast with optional confidence band."""
        fig, ax = self._fig("wide")
        ax.plot(observed.index, observed.values, color=self.theme.primary,
                lw=1, label="Observed")
        ax.plot(forecast.index, forecast.values, color=self.theme.danger,
                lw=2, ls="--", label="Forecast")
        if ci_lower is not None and ci_upper is not None:
            ax.fill_between(forecast.index, ci_lower, ci_upper,
                            color=self.theme.danger, alpha=0.15, label="95% CI")
        ax.legend()
        ax.set_title("Forecast")
        ax.set_xlabel("Date")
        return self._close(fig)

    def acf_pacf(self, series: pd.Series, lags: int = 40) -> plt.Figure:
        """Side-by-side ACF and PACF plots."""
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        fig, axes = plt.subplots(1, 2, figsize=self.theme.figsize_wide)
        plot_acf(series.dropna(), lags=lags, ax=axes[0], color=self.theme.primary)
        plot_pacf(series.dropna(), lags=lags, ax=axes[1], color=self.theme.secondary)
        axes[0].set_title("Autocorrelation (ACF)")
        axes[1].set_title("Partial Autocorrelation (PACF)")
        return self._close(fig)

    # ==================================================================
    # SAMPLING
    # ==================================================================

    def bootstrap_distribution(self, boot_result: dict) -> plt.Figure:
        """Visualize bootstrap result with CI.

        Parameters
        ----------
        boot_result : dict from ``Sampler.bootstrap_sample()``
                      Must contain ``mean``, ``std``, ``ci_95_lower``, ``ci_95_upper``.
        """
        fig, ax = self._fig()
        mu, std = boot_result["mean"], boot_result["std"]
        x = np.linspace(mu - 4 * std, mu + 4 * std, 300)
        ax.plot(x, stats.norm.pdf(x, mu, std), color=self.theme.primary, lw=2)
        ax.fill_between(x, stats.norm.pdf(x, mu, std), alpha=0.15, color=self.theme.primary)
        lo, hi = boot_result["ci_95_lower"], boot_result["ci_95_upper"]
        ax.axvline(lo, color=self.theme.danger, ls="--", label=f"95% CI [{lo:.2f}, {hi:.2f}]")
        ax.axvline(hi, color=self.theme.danger, ls="--")
        ax.axvline(mu, color=self.theme.primary, ls="-", lw=1.5, label=f"Mean = {mu:.2f}")
        ax.set_title(f"Bootstrap Distribution — {boot_result.get('column', '')}")
        ax.set_xlabel("Statistic Value")
        ax.legend()
        return self._close(fig)

    def sample_size_curve(
        self,
        population_sizes: Sequence[int],
        confidence_level: float = 0.95,
        margin_of_error: float = 0.05,
    ) -> plt.Figure:
        """Plot required sample size as a function of population size."""
        from dataspark.sampling import Sampler

        sampler = Sampler()
        sizes = [
            sampler.sample_size_calculator(N, confidence_level, margin_of_error)["required_sample_size"]
            for N in population_sizes
        ]
        fig, ax = self._fig()
        ax.plot(population_sizes, sizes, "o-", color=self.theme.primary, lw=2)
        ax.set_xlabel("Population Size")
        ax.set_ylabel("Required Sample Size")
        ax.set_title(f"Sample Size Curve (CL={confidence_level}, MoE={margin_of_error})")
        ax.grid(True, alpha=0.3)
        return self._close(fig)

    def strata_comparison(
        self,
        original: pd.DataFrame,
        sample: pd.DataFrame,
        stratum_col: str,
    ) -> plt.Figure:
        """Compare stratum proportions between original and stratified sample."""
        orig_pct = original[stratum_col].value_counts(normalize=True).sort_index()
        samp_pct = sample[stratum_col].value_counts(normalize=True).sort_index()
        fig, ax = self._fig()
        x = np.arange(len(orig_pct))
        width = 0.35
        ax.bar(x - width / 2, orig_pct.values, width, label="Population",
               color=self.theme.primary, alpha=0.7)
        ax.bar(x + width / 2, samp_pct.reindex(orig_pct.index, fill_value=0).values,
               width, label="Sample", color=self.theme.secondary, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(orig_pct.index, rotation=30, ha="right")
        ax.set_ylabel("Proportion")
        ax.set_title(f"Strata Proportions — {stratum_col}")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.legend()
        return self._close(fig)

    # ------------------------------------------------------------------
    # utility
    # ------------------------------------------------------------------
    @staticmethod
    def save(fig: plt.Figure, path: str, dpi: int = 150) -> None:
        """Save figure to file and close it."""
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def show(fig: plt.Figure) -> None:
        """Display figure (works in notebooks and scripts)."""
        fig.show()
