"""
Shared test fixtures.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """Generic mixed-type DataFrame with some missing values."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age": np.random.randint(18, 80, n).astype(float),
        "income": np.random.lognormal(10, 1, n),
        "score": np.random.normal(50, 15, n),
        "category": np.random.choice(["A", "B", "C", "D"], n),
        "gender": np.random.choice(["M", "F"], n),
        "purchased": np.random.choice([0, 1], n),
    })


@pytest.fixture
def df_with_missing(sample_df):
    """DataFrame with injected missing values."""
    df = sample_df.copy()
    rng = np.random.default_rng(42)
    for col in ["age", "income", "score"]:
        mask = rng.random(len(df)) < 0.1
        df.loc[mask, col] = np.nan
    for col in ["category", "gender"]:
        mask = rng.random(len(df)) < 0.05
        df.loc[mask, col] = None
    return df


@pytest.fixture
def df_with_outliers():
    """DataFrame with planted outliers."""
    np.random.seed(42)
    normal = np.random.normal(100, 10, 190)
    outliers = np.array([300, 400, -100, -200, 500, 600, -300, -400, 800, 1000])
    values = np.concatenate([normal, outliers])
    return pd.DataFrame({"value": values})


@pytest.fixture
def time_series():
    """Synthetic time series with trend + seasonality."""
    np.random.seed(42)
    n = 365
    t = np.arange(n)
    trend = 0.05 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 30)
    noise = np.random.normal(0, 2, n)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(100 + trend + seasonal + noise, index=dates, name="value")


@pytest.fixture
def classification_data():
    """Simple classification dataset."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=5,
        n_redundant=2, random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    return X_df, pd.Series(y, name="target")


@pytest.fixture
def regression_data():
    """Simple regression dataset."""
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=300, n_features=8, n_informative=5,
        noise=10, random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(8)])
    return X_df, pd.Series(y, name="target")
