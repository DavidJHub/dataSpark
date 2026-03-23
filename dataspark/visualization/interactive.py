"""
Interactive Explorer
====================
ipywidgets-powered interactive visualizations with sliders, dropdowns,
and toggles for parameter exploration inside Jupyter notebooks.

Each method renders a chart that re-draws live when the user drags a
slider or changes a dropdown.

Requires: ``pip install ipywidgets`` (included in ``[notebooks]`` extra).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import display

import ipywidgets as widgets

from dataspark.visualization.charts import ChartBuilder
from dataspark.visualization.themes import Theme


class InteractiveExplorer:
    """Launch interactive visualizations with parameter sliders."""

    def __init__(self, df: pd.DataFrame, theme: Theme | None = None) -> None:
        self.df = df
        self.theme = theme or Theme()
        self.theme.apply()
        self.charts = ChartBuilder(self.theme)
        self._numeric_cols = df.select_dtypes(include="number").columns.tolist()
        self._cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    # ==================================================================
    # DISTRIBUTION EXPLORER
    # ==================================================================

    def explore_distribution(self) -> widgets.VBox:
        """Interactive histogram + KDE with sliders for bins, column selection,
        and optional distribution fit overlay."""
        col_dd = widgets.Dropdown(
            options=self._numeric_cols,
            value=self._numeric_cols[0] if self._numeric_cols else None,
            description="Column:",
        )
        bins_slider = widgets.IntSlider(
            value=30, min=5, max=200, step=5, description="Bins:",
            continuous_update=False,
        )
        kde_toggle = widgets.Checkbox(value=True, description="Show KDE")
        fit_dd = widgets.Dropdown(
            options=["none", "norm", "lognorm", "expon", "gamma"],
            value="none",
            description="Fit dist:",
        )
        output = widgets.Output()

        def _update(_change=None):
            output.clear_output(wait=True)
            with output:
                fit = None if fit_dd.value == "none" else fit_dd.value
                fig = self.charts.distribution(
                    self.df[col_dd.value], bins=bins_slider.value,
                    kde=kde_toggle.value, fit_dist=fit,
                )
                plt.show()
                plt.close(fig)

        for w in [col_dd, bins_slider, kde_toggle, fit_dd]:
            w.observe(_update, names="value")
        _update()

        controls = widgets.HBox([col_dd, bins_slider, kde_toggle, fit_dd])
        return widgets.VBox([controls, output])

    # ==================================================================
    # CORRELATION EXPLORER
    # ==================================================================

    def explore_correlations(self) -> widgets.VBox:
        """Interactive correlation heatmap with method and threshold sliders."""
        method_dd = widgets.Dropdown(
            options=["pearson", "spearman", "kendall"],
            value="pearson",
            description="Method:",
        )
        threshold_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.05,
            description="Min |r|:",
            continuous_update=False,
        )
        annot_toggle = widgets.Checkbox(value=True, description="Annotations")
        output = widgets.Output()

        def _update(_change=None):
            output.clear_output(wait=True)
            with output:
                numeric = self.df.select_dtypes(include="number")
                corr = numeric.corr(method=method_dd.value)
                # Filter by threshold
                mask_thresh = corr.abs() < threshold_slider.value
                filtered = corr.copy()
                filtered[mask_thresh] = np.nan

                mask_upper = np.triu(np.ones_like(corr, dtype=bool))
                fig, ax = plt.subplots(figsize=self.theme.figsize_tall)
                sns.heatmap(
                    filtered, mask=mask_upper, annot=annot_toggle.value,
                    fmt=".2f", cmap=self.theme.diverging_cmap, center=0,
                    ax=ax, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                )
                ax.set_title(
                    f"{method_dd.value.title()} Correlations (|r| ≥ {threshold_slider.value})"
                )
                plt.tight_layout()
                plt.show()
                plt.close(fig)

        for w in [method_dd, threshold_slider, annot_toggle]:
            w.observe(_update, names="value")
        _update()

        controls = widgets.HBox([method_dd, threshold_slider, annot_toggle])
        return widgets.VBox([controls, output])

    # ==================================================================
    # OUTLIER EXPLORER
    # ==================================================================

    def explore_outliers(self) -> widgets.VBox:
        """Interactively tune outlier detection thresholds and see results."""
        col_dd = widgets.Dropdown(
            options=self._numeric_cols,
            value=self._numeric_cols[0] if self._numeric_cols else None,
            description="Column:",
        )
        method_dd = widgets.Dropdown(
            options=["iqr", "zscore", "mad"],
            value="iqr",
            description="Method:",
        )
        threshold_slider = widgets.FloatSlider(
            value=1.5, min=0.5, max=5.0, step=0.1,
            description="Threshold:",
            continuous_update=False,
        )
        output = widgets.Output()

        def _update(_change=None):
            output.clear_output(wait=True)
            with output:
                from dataspark.cleansing import OutlierDetector

                detector = OutlierDetector(
                    method=method_dd.value, threshold=threshold_slider.value
                )
                mask = detector.detect(self.df, columns=[col_dd.value])
                n_outliers = mask[col_dd.value].sum()

                fig, axes = plt.subplots(1, 2, figsize=self.theme.figsize_wide)

                # Left: scatter with outliers highlighted
                s = self.df[col_dd.value]
                normal = ~mask[col_dd.value]
                axes[0].scatter(self.df.index[normal], s[normal],
                                c=self.theme.primary, s=12, alpha=0.5, label="Normal")
                axes[0].scatter(self.df.index[mask[col_dd.value]],
                                s[mask[col_dd.value]],
                                c=self.theme.danger, s=25, alpha=0.8,
                                edgecolors="black", linewidths=0.5, label="Outlier")
                axes[0].set_title(f"Outliers: {n_outliers} / {len(self.df)}")
                axes[0].legend()

                # Right: box plot
                sns.boxplot(y=s, ax=axes[1], color=self.theme.primary)
                axes[1].set_title(f"{col_dd.value} — {method_dd.value} (t={threshold_slider.value})")

                plt.tight_layout()
                plt.show()
                plt.close(fig)

        for w in [col_dd, method_dd, threshold_slider]:
            w.observe(_update, names="value")
        _update()

        controls = widgets.HBox([col_dd, method_dd, threshold_slider])
        return widgets.VBox([controls, output])

    # ==================================================================
    # MISSING VALUE EXPLORER
    # ==================================================================

    def explore_missing(self) -> widgets.VBox:
        """Interactive missing-value analysis with imputation preview."""
        strategy_dd = widgets.Dropdown(
            options=["median", "mean", "mode", "drop", "ffill"],
            value="median",
            description="Strategy:",
        )
        col_dd = widgets.Dropdown(
            options=self._numeric_cols,
            value=self._numeric_cols[0] if self._numeric_cols else None,
            description="Preview col:",
        )
        output = widgets.Output()

        def _update(_change=None):
            output.clear_output(wait=True)
            with output:
                from dataspark.cleansing import DataCleaner

                cleaner = DataCleaner(missing_strategy=strategy_dd.value)
                cleaned = cleaner.fit_transform(self.df)

                missing_pct = (self.df.isnull().mean() * 100).sort_values(ascending=False)
                missing_pct = missing_pct[missing_pct > 0]

                fig, axes = plt.subplots(1, 2, figsize=self.theme.figsize_wide)

                # Left: missing-value bar
                if not missing_pct.empty:
                    colors = [
                        self.theme.danger if v > 30
                        else self.theme.warning if v > 10
                        else self.theme.primary
                        for v in missing_pct.values
                    ]
                    missing_pct.plot.barh(ax=axes[0], color=colors)
                    axes[0].set_xlabel("Missing %")
                    axes[0].set_title("Missing Values")
                else:
                    axes[0].text(0.5, 0.5, "No missing values", ha="center",
                                 va="center", transform=axes[0].transAxes)
                    axes[0].set_axis_off()

                # Right: before/after distribution
                col = col_dd.value
                before = self.df[col].dropna()
                after = cleaned[col].dropna() if col in cleaned.columns else before
                sns.histplot(before, kde=True, ax=axes[1], color=self.theme.neutral,
                             alpha=0.4, label="Before", stat="density")
                sns.histplot(after, kde=True, ax=axes[1], color=self.theme.success,
                             alpha=0.4, label="After", stat="density")
                axes[1].set_title(f"Imputation Preview — {col} ({strategy_dd.value})")
                axes[1].legend()

                plt.tight_layout()
                plt.show()
                plt.close(fig)

        for w in [strategy_dd, col_dd]:
            w.observe(_update, names="value")
        _update()

        controls = widgets.HBox([strategy_dd, col_dd])
        return widgets.VBox([controls, output])

    # ==================================================================
    # SCATTER / RELATIONSHIP EXPLORER
    # ==================================================================

    def explore_scatter(self) -> widgets.VBox:
        """Interactive scatter plot with axis selection, hue, size, and alpha."""
        x_dd = widgets.Dropdown(
            options=self._numeric_cols,
            value=self._numeric_cols[0] if self._numeric_cols else None,
            description="X axis:",
        )
        y_dd = widgets.Dropdown(
            options=self._numeric_cols,
            value=self._numeric_cols[1] if len(self._numeric_cols) > 1 else self._numeric_cols[0],
            description="Y axis:",
        )
        hue_dd = widgets.Dropdown(
            options=["none"] + self._cat_cols,
            value="none",
            description="Hue:",
        )
        alpha_slider = widgets.FloatSlider(
            value=0.6, min=0.05, max=1.0, step=0.05,
            description="Alpha:",
            continuous_update=False,
        )
        size_slider = widgets.IntSlider(
            value=20, min=5, max=100, step=5,
            description="Point size:",
            continuous_update=False,
        )
        reg_toggle = widgets.Checkbox(value=False, description="Regression line")
        output = widgets.Output()

        def _update(_change=None):
            output.clear_output(wait=True)
            with output:
                fig, ax = plt.subplots(figsize=self.theme.figsize)
                hue = None if hue_dd.value == "none" else hue_dd.value
                if reg_toggle.value and hue is None:
                    sns.regplot(
                        data=self.df, x=x_dd.value, y=y_dd.value, ax=ax,
                        scatter_kws={"s": size_slider.value, "alpha": alpha_slider.value,
                                     "color": self.theme.primary},
                        line_kws={"color": self.theme.danger},
                    )
                else:
                    sns.scatterplot(
                        data=self.df, x=x_dd.value, y=y_dd.value, hue=hue,
                        ax=ax, s=size_slider.value, alpha=alpha_slider.value,
                        palette=self.theme.categorical_palette[:self.df[hue].nunique()] if hue else None,
                    )
                # Annotate correlation
                r, p = stats.pearsonr(
                    self.df[x_dd.value].dropna(),
                    self.df[y_dd.value].dropna(),
                ) if len(self.df[[x_dd.value, y_dd.value]].dropna()) > 2 else (0, 1)
                ax.set_title(f"{x_dd.value} vs {y_dd.value}  (r={r:.3f}, p={p:.2e})")
                plt.tight_layout()
                plt.show()
                plt.close(fig)

        for w in [x_dd, y_dd, hue_dd, alpha_slider, size_slider, reg_toggle]:
            w.observe(_update, names="value")
        _update()

        row1 = widgets.HBox([x_dd, y_dd, hue_dd])
        row2 = widgets.HBox([alpha_slider, size_slider, reg_toggle])
        return widgets.VBox([row1, row2, output])

    # ==================================================================
    # TIME SERIES EXPLORER
    # ==================================================================

    def explore_timeseries(
        self,
        series: pd.Series,
    ) -> widgets.VBox:
        """Interactive time-series viewer with rolling-window slider
        and decomposition toggle."""
        window_slider = widgets.IntSlider(
            value=7, min=2, max=min(90, len(series) // 3), step=1,
            description="Window:",
            continuous_update=False,
        )
        show_decomp = widgets.Checkbox(value=False, description="Decomposition")
        period_slider = widgets.IntSlider(
            value=30, min=2, max=min(365, len(series) // 2), step=1,
            description="Period:",
            continuous_update=False,
        )
        output = widgets.Output()

        def _update(_change=None):
            output.clear_output(wait=True)
            with output:
                if show_decomp.value:
                    from dataspark.timeseries import TimeSeriesDecomposer

                    decomposer = TimeSeriesDecomposer(method="stl", period=period_slider.value)
                    components = decomposer.decompose(series)
                    fig = self.charts.ts_decomposition(components)
                else:
                    fig = self.charts.ts_line(
                        series,
                        rolling_window=window_slider.value,
                    )
                plt.show()
                plt.close(fig)

        for w in [window_slider, show_decomp, period_slider]:
            w.observe(_update, names="value")
        _update()

        controls = widgets.HBox([window_slider, period_slider, show_decomp])
        return widgets.VBox([controls, output])

    # ==================================================================
    # SAMPLING EXPLORER
    # ==================================================================

    def explore_sampling(self) -> widgets.VBox:
        """Explore how sample size and method affect representativeness."""
        if not self._cat_cols:
            out = widgets.Output()
            with out:
                print("No categorical columns found for stratified sampling exploration.")
            return widgets.VBox([out])

        stratum_dd = widgets.Dropdown(
            options=self._cat_cols,
            value=self._cat_cols[0],
            description="Stratum:",
        )
        frac_slider = widgets.FloatSlider(
            value=0.3, min=0.05, max=0.95, step=0.05,
            description="Frac:",
            continuous_update=False,
        )
        method_dd = widgets.Dropdown(
            options=["stratified", "systematic", "cluster"],
            value="stratified",
            description="Method:",
        )
        output = widgets.Output()

        def _update(_change=None):
            output.clear_output(wait=True)
            with output:
                from dataspark.sampling import Sampler

                sampler = Sampler()
                col = stratum_dd.value
                if method_dd.value == "stratified":
                    sample = sampler.stratified_sample(self.df, col, frac=frac_slider.value)
                elif method_dd.value == "systematic":
                    k = max(2, int(1 / frac_slider.value))
                    sample = sampler.systematic_sample(self.df, k=k)
                else:  # cluster
                    n_clusters = max(1, int(self.df[col].nunique() * frac_slider.value))
                    sample = sampler.cluster_sample(self.df, col, n_clusters=n_clusters)

                fig = self.charts.strata_comparison(self.df, sample, col)
                fig.suptitle(
                    f"{method_dd.value.title()} Sampling — {len(sample)}/{len(self.df)} rows "
                    f"(frac≈{frac_slider.value})",
                    fontsize=self.theme.title_size,
                )
                plt.tight_layout()
                plt.show()
                plt.close(fig)

        for w in [stratum_dd, frac_slider, method_dd]:
            w.observe(_update, names="value")
        _update()

        controls = widgets.HBox([stratum_dd, method_dd, frac_slider])
        return widgets.VBox([controls, output])

    # ==================================================================
    # STATISTICAL TEST EXPLORER
    # ==================================================================

    def explore_hypothesis_test(self) -> widgets.VBox:
        """Compare two numeric columns with selectable parametric /
        non-parametric tests and adjustable sample sizes."""
        if len(self._numeric_cols) < 2:
            out = widgets.Output()
            with out:
                print("Need at least 2 numeric columns for hypothesis testing.")
            return widgets.VBox([out])

        col_a_dd = widgets.Dropdown(
            options=self._numeric_cols,
            value=self._numeric_cols[0],
            description="Group A:",
        )
        col_b_dd = widgets.Dropdown(
            options=self._numeric_cols,
            value=self._numeric_cols[1],
            description="Group B:",
        )
        test_dd = widgets.Dropdown(
            options=["welch_t", "mann_whitney", "ks_test"],
            value="welch_t",
            description="Test:",
        )
        subsample_slider = widgets.IntSlider(
            value=min(len(self.df), 200),
            min=10,
            max=len(self.df),
            step=10,
            description="N samples:",
            continuous_update=False,
        )
        output = widgets.Output()

        def _update(_change=None):
            output.clear_output(wait=True)
            with output:
                from dataspark.statistical import HypothesisTester, NonParametricTests
                from dataspark.statistical.effect_size import EffectSizeCalculator

                n = subsample_slider.value
                a = self.df[col_a_dd.value].dropna().sample(
                    n=min(n, len(self.df[col_a_dd.value].dropna())), random_state=42
                ).values
                b = self.df[col_b_dd.value].dropna().sample(
                    n=min(n, len(self.df[col_b_dd.value].dropna())), random_state=42
                ).values

                if test_dd.value == "welch_t":
                    result = HypothesisTester.t_test(a, b)
                elif test_dd.value == "mann_whitney":
                    result = NonParametricTests.mann_whitney(a, b)
                else:
                    result = NonParametricTests.ks_two_sample(a, b)

                es = EffectSizeCalculator.cohens_d(a, b)

                fig, axes = plt.subplots(1, 2, figsize=self.theme.figsize_wide)

                # Left: overlapping distributions
                sns.histplot(a, kde=True, ax=axes[0], color=self.theme.primary,
                             alpha=0.4, label=col_a_dd.value, stat="density")
                sns.histplot(b, kde=True, ax=axes[0], color=self.theme.secondary,
                             alpha=0.4, label=col_b_dd.value, stat="density")
                sig = "★ Significant" if result["p_value"] < 0.05 else "Not Significant"
                axes[0].set_title(
                    f'{result["test"]}  p={result["p_value"]:.4f}  {sig}'
                )
                axes[0].legend()

                # Right: effect size gauge
                d_val = abs(es["cohens_d"])
                color = (
                    self.theme.success if d_val < 0.2
                    else self.theme.warning if d_val < 0.5
                    else self.theme.danger
                )
                axes[1].barh(["Cohen's d"], [d_val], color=color, height=0.3)
                for thresh, lbl in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
                    axes[1].axvline(thresh, color=self.theme.neutral, ls=":", lw=0.8)
                    axes[1].text(thresh, 0.35, lbl, fontsize=8, ha="center",
                                 color=self.theme.neutral)
                axes[1].set_title(f"Effect Size: {es['cohens_d']:.3f} ({es['magnitude']})")
                axes[1].set_xlabel("|d|")

                plt.tight_layout()
                plt.show()
                plt.close(fig)

        for w in [col_a_dd, col_b_dd, test_dd, subsample_slider]:
            w.observe(_update, names="value")
        _update()

        row1 = widgets.HBox([col_a_dd, col_b_dd])
        row2 = widgets.HBox([test_dd, subsample_slider])
        return widgets.VBox([row1, row2, output])

    # ==================================================================
    # CONVENIENCE: Launch all
    # ==================================================================

    def launch(self) -> widgets.Tab:
        """Launch a tabbed interface with all interactive explorers."""
        tab_contents = {
            "Distribution": self.explore_distribution(),
            "Correlation": self.explore_correlations(),
            "Scatter": self.explore_scatter(),
            "Outliers": self.explore_outliers(),
            "Missing": self.explore_missing(),
        }
        if self._cat_cols:
            tab_contents["Sampling"] = self.explore_sampling()
        if len(self._numeric_cols) >= 2:
            tab_contents["Hypothesis"] = self.explore_hypothesis_test()

        tab = widgets.Tab(children=list(tab_contents.values()))
        for i, title in enumerate(tab_contents.keys()):
            tab.set_title(i, title)
        return tab
