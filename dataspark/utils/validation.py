"""
Input Validation Utilities
==========================
Guard functions for DataFrame inputs across all modules.
"""

from __future__ import annotations

import pandas as pd


def validate_dataframe(obj: object, min_rows: int = 1) -> pd.DataFrame:
    """Ensure the input is a non-empty DataFrame."""
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(obj).__name__}")
    if len(obj) < min_rows:
        raise ValueError(f"DataFrame must have at least {min_rows} rows, got {len(obj)}")
    return obj


def validate_column_exists(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found. Available: {df.columns.tolist()}")


def validate_numeric_column(df: pd.DataFrame, column: str) -> None:
    validate_column_exists(df, column)
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"Column '{column}' is not numeric (dtype: {df[column].dtype})")
