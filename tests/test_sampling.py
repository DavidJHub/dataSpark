"""Tests for dataspark.sampling module."""

import numpy as np
import pandas as pd
import pytest

from dataspark.sampling import Sampler


class TestSampler:
    def test_stratified_sample_n(self, sample_df):
        sampler = Sampler()
        result = sampler.stratified_sample(sample_df, stratum_col="category", n=50)
        assert len(result) <= 50

    def test_stratified_sample_frac(self, sample_df):
        sampler = Sampler()
        result = sampler.stratified_sample(sample_df, stratum_col="category", frac=0.3)
        assert len(result) < len(sample_df)

    def test_cluster_sample(self, sample_df):
        sampler = Sampler()
        result = sampler.cluster_sample(sample_df, cluster_col="category", n_clusters=2)
        assert result["category"].nunique() <= 2

    def test_systematic_sample(self, sample_df):
        sampler = Sampler()
        result = sampler.systematic_sample(sample_df, k=5)
        assert len(result) == pytest.approx(len(sample_df) / 5, abs=1)

    def test_reservoir_sample(self, sample_df):
        sampler = Sampler()
        result = sampler.reservoir_sample(sample_df, n=30)
        assert len(result) == 30

    def test_bootstrap(self, sample_df):
        sampler = Sampler()
        result = sampler.bootstrap_sample(sample_df, n_samples=500, column="income")
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] < result["ci_upper"]

    def test_sample_size_calculator(self):
        sampler = Sampler()
        result = sampler.sample_size_calculator(population_size=10000)
        assert result["required_sample_size"] > 0
        assert result["required_sample_size"] < 10000
