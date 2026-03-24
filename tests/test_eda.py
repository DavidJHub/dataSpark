"""Tests for dataspark.eda module."""

import numpy as np
import pandas as pd
import pytest

from dataspark.eda import DataExplorer, CorrelationAnalyzer, DistributionAnalyzer


class TestDataExplorer:
    def test_summary(self, sample_df):
        explorer = DataExplorer(sample_df)
        result = explorer.summary()
        assert "skewness" in result.columns
        assert "kurtosis" in result.columns
        assert "cv" in result.columns

    def test_categorical_summary(self, sample_df):
        explorer = DataExplorer(sample_df)
        result = explorer.categorical_summary()
        assert "category" in result
        assert "count" in result["category"].columns

    def test_normality_tests(self, sample_df):
        explorer = DataExplorer(sample_df)
        result = explorer.normality_tests()
        assert "p_value" in result.columns
        assert "is_normal" in result.columns

    def test_info_report(self, sample_df):
        explorer = DataExplorer(sample_df)
        info = explorer.info_report()
        assert "shape" in info
        assert "memory_mb" in info
        assert "total_missing" in info


class TestCorrelationAnalyzer:
    def test_correlation_matrix(self, sample_df):
        analyzer = CorrelationAnalyzer(sample_df)
        corr = analyzer.correlation_matrix("pearson")
        assert corr.shape[0] == corr.shape[1]

    def test_pairwise_significance(self, sample_df):
        analyzer = CorrelationAnalyzer(sample_df)
        result = analyzer.pairwise_significance()
        assert "p_value" in result.columns
        assert "significant" in result.columns
        assert "strength" in result.columns

    def test_top_correlations(self, sample_df):
        analyzer = CorrelationAnalyzer(sample_df)
        top = analyzer.top_correlations(n=5)
        assert len(top) <= 5
        assert "correlation" in top.columns


class TestDistributionAnalyzer:
    def test_fit_distributions(self, sample_df):
        analyzer = DistributionAnalyzer(sample_df)
        result = analyzer.fit_distributions("score")
        assert "distribution" in result.columns
        assert "bic" in result.columns
        assert len(result) > 0

    def test_detect_multimodality(self, sample_df):
        analyzer = DistributionAnalyzer(sample_df)
        result = analyzer.detect_multimodality("score")
        assert "bimodality_coefficient" in result
        assert "is_multimodal" in result

    def test_quantile_analysis(self, sample_df):
        analyzer = DistributionAnalyzer(sample_df)
        result = analyzer.quantile_analysis("score")
        assert "quantiles" in result
        assert "iqr" in result
