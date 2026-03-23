"""Tests for dataspark.cleansing module."""

import numpy as np
import pandas as pd
import pytest

from dataspark.cleansing import DataCleaner, OutlierDetector, TypeInferenceEngine, Deduplicator


class TestDataCleaner:
    def test_fit_transform_median(self, df_with_missing):
        cleaner = DataCleaner(missing_strategy="median")
        result = cleaner.fit_transform(df_with_missing)
        assert result.select_dtypes(include="number").isnull().sum().sum() == 0

    def test_fit_transform_mean(self, df_with_missing):
        cleaner = DataCleaner(missing_strategy="mean")
        result = cleaner.fit_transform(df_with_missing)
        assert result.select_dtypes(include="number").isnull().sum().sum() == 0

    def test_fit_transform_drop(self, df_with_missing):
        cleaner = DataCleaner(missing_strategy="drop")
        result = cleaner.fit_transform(df_with_missing)
        assert result.isnull().sum().sum() == 0
        assert len(result) < len(df_with_missing)

    def test_fit_transform_ffill(self, df_with_missing):
        cleaner = DataCleaner(missing_strategy="ffill")
        result = cleaner.fit_transform(df_with_missing)
        numeric_nulls = result.select_dtypes(include="number").isnull().sum().sum()
        assert numeric_nulls == 0

    def test_fit_transform_knn(self, df_with_missing):
        cleaner = DataCleaner(missing_strategy="knn", knn_neighbors=3)
        result = cleaner.fit_transform(df_with_missing)
        assert result.select_dtypes(include="number").isnull().sum().sum() == 0

    def test_standardize_columns(self):
        df = pd.DataFrame({"  First Name  ": [1], "Last-Name": [2], "AGE (years)": [3]})
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df)
        assert all(c == c.lower() for c in result.columns)
        assert " " not in "".join(result.columns)

    def test_strip_whitespace(self):
        df = pd.DataFrame({"name": ["  Alice  ", "  Bob  "], "val": [1, 2]})
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df)
        assert result["name"].iloc[0] == "Alice"

    def test_profile_missing(self, df_with_missing):
        cleaner = DataCleaner()
        profile = cleaner.profile_missing(df_with_missing)
        assert "missing_count" in profile.columns
        assert "missing_pct" in profile.columns
        assert len(profile) > 0


class TestOutlierDetector:
    def test_iqr_detection(self, df_with_outliers):
        detector = OutlierDetector(method="iqr")
        mask = detector.detect(df_with_outliers)
        assert mask["value"].sum() > 0

    def test_zscore_detection(self, df_with_outliers):
        detector = OutlierDetector(method="zscore", threshold=3.0)
        mask = detector.detect(df_with_outliers)
        assert mask["value"].sum() > 0

    def test_mad_detection(self, df_with_outliers):
        detector = OutlierDetector(method="mad", threshold=3.5)
        mask = detector.detect(df_with_outliers)
        assert mask["value"].sum() > 0

    def test_remove(self, df_with_outliers):
        detector = OutlierDetector(method="iqr")
        result = detector.remove(df_with_outliers)
        assert len(result) < len(df_with_outliers)

    def test_cap(self, df_with_outliers):
        detector = OutlierDetector(method="iqr")
        result = detector.cap(df_with_outliers)
        assert len(result) == len(df_with_outliers)
        assert result["value"].max() < df_with_outliers["value"].max()


class TestTypeInferenceEngine:
    def test_infer_and_convert(self):
        df = pd.DataFrame({
            "num_str": ["1", "2", "3", "4", "5"],
            "date_str": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"],
            "bool_str": ["true", "false", "true", "false", "true"],
            "cat_col": ["A", "A", "A", "B", "A"],
        })
        engine = TypeInferenceEngine()
        result = engine.infer_and_convert(df)
        assert pd.api.types.is_numeric_dtype(result["num_str"])
        assert pd.api.types.is_datetime64_any_dtype(result["date_str"])

    def test_report(self, sample_df):
        engine = TypeInferenceEngine()
        report = engine.report(sample_df)
        assert "current_dtype" in report.columns
        assert "suggested_dtype" in report.columns


class TestDeduplicator:
    def test_exact_dedup(self):
        df = pd.DataFrame({"a": [1, 2, 2, 3], "b": ["x", "y", "y", "z"]})
        dedup = Deduplicator(strategy="exact")
        result = dedup.deduplicate(df)
        assert len(result) == 3

    def test_fuzzy_dedup(self):
        df = pd.DataFrame({"name": ["John Smith", "Jon Smith", "Jane Doe", "John Smithh"]})
        dedup = Deduplicator(strategy="fuzzy", similarity_threshold=0.7)
        dups = dedup.find_duplicates(df)
        assert len(dups) > 0

    def test_hash_rows(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        dedup = Deduplicator()
        hashes = dedup.hash_rows(df)
        assert len(hashes) == 3
        assert hashes.nunique() == 3
