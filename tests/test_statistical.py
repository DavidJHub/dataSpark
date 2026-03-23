"""Tests for dataspark.statistical module."""

import numpy as np
import pandas as pd
import pytest

from dataspark.statistical import HypothesisTester, NonParametricTests, EffectSizeCalculator


class TestHypothesisTester:
    def test_t_test(self):
        np.random.seed(42)
        a = np.random.normal(10, 2, 100)
        b = np.random.normal(12, 2, 100)
        result = HypothesisTester.t_test(a, b)
        assert result["p_value"] < 0.05
        assert "statistic" in result

    def test_paired_t_test(self):
        np.random.seed(42)
        before = np.random.normal(50, 10, 50)
        after = before + np.random.normal(5, 3, 50)
        result = HypothesisTester.paired_t_test(before, after)
        assert result["p_value"] < 0.05

    def test_one_way_anova(self):
        np.random.seed(42)
        g1 = np.random.normal(10, 2, 50)
        g2 = np.random.normal(12, 2, 50)
        g3 = np.random.normal(14, 2, 50)
        result = HypothesisTester.one_way_anova(g1, g2, g3)
        assert result["p_value"] < 0.05
        assert result["n_groups"] == 3

    def test_chi_squared(self):
        table = pd.DataFrame({"yes": [30, 10], "no": [20, 40]})
        result = HypothesisTester.chi_squared(table)
        assert result["p_value"] < 0.05

    def test_proportion_z_test(self):
        result = HypothesisTester.proportion_z_test(80, 100, 60, 100)
        assert result["p_value"] < 0.05


class TestNonParametricTests:
    def test_mann_whitney(self):
        np.random.seed(42)
        a = np.random.exponential(5, 50)
        b = np.random.exponential(8, 50)
        result = NonParametricTests.mann_whitney(a, b)
        assert "p_value" in result

    def test_wilcoxon(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        y = x + np.random.normal(2, 0.5, 50)
        result = NonParametricTests.wilcoxon_signed_rank(x, y)
        assert result["p_value"] < 0.05

    def test_kruskal_wallis(self):
        np.random.seed(42)
        g1 = np.random.exponential(3, 40)
        g2 = np.random.exponential(6, 40)
        g3 = np.random.exponential(9, 40)
        result = NonParametricTests.kruskal_wallis(g1, g2, g3)
        assert result["n_groups"] == 3

    def test_ks_two_sample(self):
        np.random.seed(42)
        a = np.random.normal(0, 1, 100)
        b = np.random.normal(1, 1, 100)
        result = NonParametricTests.ks_two_sample(a, b)
        assert result["p_value"] < 0.05

    def test_runs_test(self):
        np.random.seed(42)
        series = pd.Series(np.random.normal(0, 1, 100))
        result = NonParametricTests.runs_test(series)
        assert "n_runs" in result


class TestEffectSizeCalculator:
    def test_cohens_d(self):
        a = np.array([10, 12, 14, 11, 13])
        b = np.array([20, 22, 24, 21, 23])
        result = EffectSizeCalculator.cohens_d(a, b)
        assert result["magnitude"] == "large"

    def test_power_analysis(self):
        result = EffectSizeCalculator.power_analysis(0.5, 100)
        assert 0 <= result["power"] <= 1
