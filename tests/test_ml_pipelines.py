"""Tests for dataspark.ml_pipelines module."""

import numpy as np
import pandas as pd
import pytest

from dataspark.ml_pipelines import PipelineBuilder, ModelSelector, FeatureEngineer


class TestPipelineBuilder:
    def test_build_classification(self, classification_data):
        X, y = classification_data
        builder = PipelineBuilder(task="classification")
        pipe = builder.build(X, model_name="logistic_regression")
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)

    def test_build_regression(self, regression_data):
        X, y = regression_data
        builder = PipelineBuilder(task="regression")
        pipe = builder.build(X, model_name="ridge")
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)

    def test_cross_validate(self, classification_data):
        X, y = classification_data
        builder = PipelineBuilder(task="classification")
        pipe = builder.build(X, model_name="random_forest")
        results = builder.cross_validate(pipe, X, y, cv=3)
        assert "test_accuracy_mean" in results
        assert results["test_accuracy_mean"] > 0.5

    def test_mixed_dtypes(self):
        """Pipeline handles mixed numeric + categorical data."""
        X = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
            "cat1": ["a", "b", "a", "b", "a"] * 20,
        })
        y = pd.Series([0, 1, 0, 1, 0] * 20)
        builder = PipelineBuilder(task="classification")
        pipe = builder.build(X)
        pipe.fit(X, y)
        assert len(pipe.predict(X)) == len(y)


class TestModelSelector:
    def test_compare_models(self, classification_data):
        X, y = classification_data
        selector = ModelSelector(task="classification", cv=3)
        results = selector.compare_models(X, y)
        assert "model" in results.columns
        assert "test_score_mean" in results.columns
        assert len(results) > 1


class TestFeatureEngineer:
    def test_interaction_features(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = FeatureEngineer.create_interaction_features(df, ["a", "b"])
        assert "a_x_b" in result.columns

    def test_polynomial_features(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = FeatureEngineer.create_polynomial_features(df, ["x"], degree=3)
        assert "x_pow2" in result.columns
        assert "x_pow3" in result.columns

    def test_log_features(self):
        df = pd.DataFrame({"x": [1, 10, 100]})
        result = FeatureEngineer.create_log_features(df, ["x"])
        assert "x_log" in result.columns

    def test_select_k_best(self, classification_data):
        X, y = classification_data
        scores = FeatureEngineer.select_k_best(X, y, k=5)
        assert len(scores) == X.shape[1]
        assert "score" in scores.columns

    def test_pca_reduce(self, classification_data):
        X, y = classification_data
        transformed, pca = FeatureEngineer.pca_reduce(X, n_components=3)
        assert transformed.shape[1] == 3
