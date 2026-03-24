"""Automatic pandas dtype inference and conversion utilities.

This module provides :class:`TypeInferenceEngine`, which tries to convert
columns to more appropriate and memory-efficient dtypes while preserving data
quality as much as possible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class TypeInferenceEngine:
    """Infer and convert DataFrame columns to optimized pandas dtypes.

    Parameters
    ----------
    categorical_threshold:
        Maximum unique ratio (``nunique / n_rows``) to cast string columns to
        ``category`` dtype.
    datetime_formats:
        Candidate datetime formats attempted in order for string columns.
    """

    def __init__(
        self,
        categorical_threshold: float = 0.05,
        datetime_formats: list[str] | None = None,
    ) -> None:
        """Initialize type inference configuration."""
        self.categorical_threshold = categorical_threshold
        self.datetime_formats = datetime_formats or [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y",
        ]

    def infer_and_convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infer optimal dtypes and return a converted dataframe copy.

        Logs memory usage before/after conversion.
        """
        df = df.copy()
        before_mem = df.memory_usage(deep=True).sum()
        for col in df.columns:
            df[col] = self._convert_column(df[col])
        after_mem = df.memory_usage(deep=True).sum()
        reduction = (1 - after_mem / before_mem) * 100 if before_mem > 0 else 0
        logger.info(
            "Memory reduced by {:.1f}% ({:.1f}MB → {:.1f}MB)",
            reduction,
            before_mem / 1e6,
            after_mem / 1e6,
        )
        return df

    def report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a report with current vs suggested dtype per column."""
        rows = []
        for col in df.columns:
            suggested = self._suggest_dtype(df[col])
            rows.append(
                {
                    "column": col,
                    "current_dtype": str(df[col].dtype),
                    "suggested_dtype": suggested,
                    "nunique": df[col].nunique(),
                    "null_pct": df[col].isnull().mean() * 100,
                }
            )
        return pd.DataFrame(rows)

    def _convert_column(self, s: pd.Series) -> pd.Series:
        """Attempt conversion of a single series following heuristic order.

        Conversion order:

        1. boolean-like values,
        2. datetime strings,
        3. numeric strings,
        4. numeric downcast,
        5. categorical strings.
        """
        unique_vals = set(s.dropna().unique())
        if unique_vals <= {0, 1, True, False, "true", "false", "True", "False", "0", "1"}:
            try:
                return s.map(
                    {
                        "true": True,
                        "false": False,
                        "True": True,
                        "False": False,
                        "1": True,
                        "0": False,
                        1: True,
                        0: False,
                    }
                ).astype("boolean")
            except (ValueError, TypeError):
                pass

        is_string = pd.api.types.is_string_dtype(s) or s.dtype == "object"

        if is_string:
            for fmt in self.datetime_formats:
                try:
                    converted = pd.to_datetime(s, format=fmt, errors="coerce")
                    if converted.notna().mean() > 0.8:
                        return converted
                except Exception:
                    continue

        if is_string:
            numeric = pd.to_numeric(s, errors="coerce")
            if numeric.notna().mean() > 0.8:
                return self._downcast_numeric(numeric)

        if pd.api.types.is_numeric_dtype(s):
            return self._downcast_numeric(s)

        if is_string and len(s) > 0:
            ratio = s.nunique() / len(s)
            if ratio < self.categorical_threshold:
                return s.astype("category")

        return s

    def _suggest_dtype(self, s: pd.Series) -> str:
        """Return dtype name resulting from :meth:`_convert_column`."""
        converted = self._convert_column(s)
        return str(converted.dtype)

    @staticmethod
    def _downcast_numeric(s: pd.Series) -> pd.Series:
        """Downcast float/integer series to smaller numeric dtype when possible."""
        if pd.api.types.is_float_dtype(s):
            return pd.to_numeric(s, downcast="float")
        if pd.api.types.is_integer_dtype(s):
            return pd.to_numeric(s, downcast="integer")
        return s
