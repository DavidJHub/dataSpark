"""Tests for dataspark.visualization module."""

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from dataspark.visualization.charts import ChartBuilder
from dataspark.visualization.dashboard import Dashboard
from dataspark.visualization.themes import Theme


@pytest.fixture(autouse=True)
def _close_figs():
    """Ensure all matplotlib figures are closed after each test."""
    yield
    plt.close("all")


class TestTheme:
    def test_default(self):
        t = Theme()
        assert t.primary == "#2563EB"
        t.apply()

    def test_dark(self):
        t = Theme.dark()
        assert t.style == "darkgrid"

    def test_minimal(self):
        t = Theme.minimal()
        assert t.style == "ticks"


class TestChartBuilder:
    # -- Data profiling --
    def test_missing_matrix(self, df_with_missing):
        cb = ChartBuilder()
        fig = cb.missing_matrix(df_with_missing)
        assert isinstance(fig, plt.Figure)

    def test_missing_bar(self, df_with_missing):
        cb = ChartBuilder()
        fig = cb.missing_bar(df_with_missing)
        assert isinstance(fig, plt.Figure)

    def test_missing_bar_no_missing(self, sample_df):
        cb = ChartBuilder()
        fig = cb.missing_bar(sample_df)
        assert isinstance(fig, plt.Figure)

    def test_outlier_scatter(self, df_with_outliers):
        from dataspark.cleansing import OutlierDetector
        detector = OutlierDetector(method="iqr")
        mask = detector.detect(df_with_outliers)["value"]
        cb = ChartBuilder()
        fig = cb.outlier_scatter(df_with_outliers, "value", mask)
        assert isinstance(fig, plt.Figure)

    def test_before_after_cleaning(self, df_with_missing):
        from dataspark.cleansing import DataCleaner
        cleaner = DataCleaner(missing_strategy="median")
        cleaned = cleaner.fit_transform(df_with_missing)
        cb = ChartBuilder()
        fig = cb.before_after_cleaning(df_with_missing, cleaned, "age")
        assert isinstance(fig, plt.Figure)

    # -- EDA --
    def test_distribution(self, sample_df):
        cb = ChartBuilder()
        fig = cb.distribution(sample_df["score"], bins=20)
        assert isinstance(fig, plt.Figure)

    def test_distribution_with_fit(self, sample_df):
        cb = ChartBuilder()
        fig = cb.distribution(sample_df["score"], fit_dist="norm")
        assert isinstance(fig, plt.Figure)

    def test_correlation_heatmap(self, sample_df):
        cb = ChartBuilder()
        fig = cb.correlation_heatmap(sample_df)
        assert isinstance(fig, plt.Figure)

    def test_top_correlations_bar(self, sample_df):
        from dataspark.eda import CorrelationAnalyzer
        analyzer = CorrelationAnalyzer(sample_df)
        top = analyzer.top_correlations(n=5)
        cb = ChartBuilder()
        fig = cb.top_correlations_bar(top)
        assert isinstance(fig, plt.Figure)

    def test_categorical_bars(self, sample_df):
        cb = ChartBuilder()
        fig = cb.categorical_bars(sample_df, "category")
        assert isinstance(fig, plt.Figure)

    def test_categorical_bars_with_hue(self, sample_df):
        cb = ChartBuilder()
        fig = cb.categorical_bars(sample_df, "category", hue="gender")
        assert isinstance(fig, plt.Figure)

    def test_qq_plot(self, sample_df):
        cb = ChartBuilder()
        fig = cb.qq_plot(sample_df["score"])
        assert isinstance(fig, plt.Figure)

    # -- Statistical --
    def test_p_value_forest(self):
        results = [
            {"test": "t-test A vs B", "p_value": 0.03},
            {"test": "ANOVA groups", "p_value": 0.12},
            {"test": "chi-sq", "p_value": 0.001},
        ]
        cb = ChartBuilder()
        fig = cb.p_value_forest(results)
        assert isinstance(fig, plt.Figure)

    def test_effect_size_bar(self):
        results = [
            {"label": "A vs B", "value": 0.8},
            {"label": "C vs D", "value": 0.3},
        ]
        cb = ChartBuilder()
        fig = cb.effect_size_bar(results)
        assert isinstance(fig, plt.Figure)

    def test_group_comparison(self):
        groups = {
            "Control": np.random.normal(10, 2, 50),
            "Treatment": np.random.normal(12, 2, 50),
        }
        cb = ChartBuilder()
        for kind in ["box", "violin", "strip"]:
            fig = cb.group_comparison(groups, kind=kind)
            assert isinstance(fig, plt.Figure)

    # -- ML --
    def test_model_comparison(self):
        results = pd.DataFrame({
            "model": ["RF", "LR", "GBM"],
            "train_score_mean": [0.95, 0.88, 0.93],
            "test_score_mean": [0.90, 0.86, 0.91],
            "test_score_std": [0.02, 0.03, 0.02],
        })
        cb = ChartBuilder()
        fig = cb.model_comparison(results)
        assert isinstance(fig, plt.Figure)

    def test_feature_importance(self):
        scores = pd.DataFrame({
            "feature": [f"f{i}" for i in range(10)],
            "score": np.random.rand(10),
        })
        cb = ChartBuilder()
        fig = cb.feature_importance(scores)
        assert isinstance(fig, plt.Figure)

    def test_confusion_matrix(self):
        y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 1])
        cb = ChartBuilder()
        fig = cb.confusion_matrix(y_true, y_pred)
        assert isinstance(fig, plt.Figure)

    def test_pca_variance(self):
        cb = ChartBuilder()
        fig = cb.pca_variance(np.array([0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02]))
        assert isinstance(fig, plt.Figure)

    def test_residuals(self):
        y_true = np.random.normal(100, 10, 100)
        y_pred = y_true + np.random.normal(0, 5, 100)
        cb = ChartBuilder()
        fig = cb.residuals(y_true, y_pred)
        assert isinstance(fig, plt.Figure)

    # -- Time series --
    def test_ts_line(self, time_series):
        cb = ChartBuilder()
        fig = cb.ts_line(time_series, rolling_window=14)
        assert isinstance(fig, plt.Figure)

    def test_ts_decomposition(self, time_series):
        from dataspark.timeseries import TimeSeriesDecomposer
        decomposer = TimeSeriesDecomposer(method="stl", period=30)
        components = decomposer.decompose(time_series)
        cb = ChartBuilder()
        fig = cb.ts_decomposition(components)
        assert isinstance(fig, plt.Figure)

    def test_ts_forecast(self, time_series):
        train = time_series[:300]
        future_idx = time_series.index[300:]
        forecast = pd.Series(
            np.full(len(future_idx), train.mean()), index=future_idx
        )
        cb = ChartBuilder()
        fig = cb.ts_forecast(train, forecast)
        assert isinstance(fig, plt.Figure)

    def test_acf_pacf(self, time_series):
        cb = ChartBuilder()
        fig = cb.acf_pacf(time_series, lags=20)
        assert isinstance(fig, plt.Figure)

    # -- Sampling --
    def test_bootstrap_distribution(self, sample_df):
        from dataspark.sampling import Sampler
        sampler = Sampler()
        boot = sampler.bootstrap_sample(sample_df, column="income", n_samples=500)
        cb = ChartBuilder()
        fig = cb.bootstrap_distribution(boot)
        assert isinstance(fig, plt.Figure)

    def test_sample_size_curve(self):
        cb = ChartBuilder()
        fig = cb.sample_size_curve([100, 500, 1000, 5000, 10000, 50000])
        assert isinstance(fig, plt.Figure)

    def test_strata_comparison(self, sample_df):
        from dataspark.sampling import Sampler
        sampler = Sampler()
        sample = sampler.stratified_sample(sample_df, "category", frac=0.3)
        cb = ChartBuilder()
        fig = cb.strata_comparison(sample_df, sample, "category")
        assert isinstance(fig, plt.Figure)

    # -- Utility --
    def test_save(self, sample_df, tmp_path):
        cb = ChartBuilder()
        fig = cb.distribution(sample_df["score"])
        path = str(tmp_path / "test.png")
        cb.save(fig, path)
        import os
        assert os.path.exists(path)


class TestDashboard:
    def test_data_quality(self, df_with_missing):
        dash = Dashboard()
        fig = dash.data_quality(df_with_missing)
        assert isinstance(fig, plt.Figure)

    def test_eda_overview(self, sample_df):
        dash = Dashboard()
        fig = dash.eda_overview(sample_df)
        assert isinstance(fig, plt.Figure)

    def test_model_report(self):
        comparison = pd.DataFrame({
            "model": ["RF", "LR"],
            "train_score_mean": [0.95, 0.88],
            "test_score_mean": [0.90, 0.86],
            "test_score_std": [0.02, 0.03],
            "overfit_gap": [0.05, 0.02],
        })
        scores = pd.DataFrame({
            "feature": [f"f{i}" for i in range(5)],
            "score": np.random.rand(5),
        })
        dash = Dashboard()
        fig = dash.model_report(comparison, feature_scores=scores)
        assert isinstance(fig, plt.Figure)

    def test_model_report_with_predictions(self):
        comparison = pd.DataFrame({
            "model": ["RF"],
            "train_score_mean": [0.95],
            "test_score_mean": [0.90],
            "test_score_std": [0.02],
            "overfit_gap": [0.05],
        })
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        dash = Dashboard()
        fig = dash.model_report(comparison, y_true=y_true, y_pred=y_pred)
        assert isinstance(fig, plt.Figure)

    def test_timeseries_report(self, time_series):
        from dataspark.timeseries import TimeSeriesDecomposer
        decomposer = TimeSeriesDecomposer(method="stl", period=30)
        components = decomposer.decompose(time_series)
        dash = Dashboard()
        fig = dash.timeseries_report(time_series, components=components)
        assert isinstance(fig, plt.Figure)
