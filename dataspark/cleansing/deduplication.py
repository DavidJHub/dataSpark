"""
Deduplication Engine
====================
Exact and fuzzy duplicate detection for large-scale tabular data.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger


class Deduplicator:
    """Detect and remove duplicate records — exact or fuzzy."""

    def __init__(
        self,
        strategy: Literal["exact", "fuzzy"] = "exact",
        similarity_threshold: float = 0.85,
        subset: list[str] | None = None,
    ) -> None:
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
        self.subset = subset

    def find_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return duplicate rows (keeps first occurrence marked as non-duplicate)."""
        if self.strategy == "exact":
            mask = df.duplicated(subset=self.subset, keep="first")
        else:
            mask = self._fuzzy_duplicates(df)
        n = mask.sum()
        logger.info("Found {} duplicate rows (strategy={})", n, self.strategy)
        return df[mask]

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates and return cleaned DataFrame."""
        if self.strategy == "exact":
            result = df.drop_duplicates(subset=self.subset, keep="first").reset_index(drop=True)
        else:
            mask = self._fuzzy_duplicates(df)
            result = df[~mask].reset_index(drop=True)
        removed = len(df) - len(result)
        logger.info("Removed {} duplicates — {} → {} rows", removed, len(df), len(result))
        return result

    def hash_rows(self, df: pd.DataFrame) -> pd.Series:
        """Compute a SHA-256 hash for each row (useful for large-scale dedup)."""
        cols = self.subset or df.columns.tolist()
        return df[cols].apply(
            lambda row: hashlib.sha256(str(row.values).encode()).hexdigest(), axis=1
        )

    def _fuzzy_duplicates(self, df: pd.DataFrame) -> pd.Series:
        """Simple character-level similarity for string columns."""
        cols = self.subset or df.select_dtypes(include="object").columns.tolist()
        if not cols:
            return df.duplicated(keep="first")

        mask = pd.Series(False, index=df.index)
        seen: list[str] = []

        for idx, row in df.iterrows():
            row_str = " ".join(str(row[c]) for c in cols)
            is_dup = False
            for s in seen:
                if self._char_similarity(row_str, s) >= self.similarity_threshold:
                    is_dup = True
                    break
            mask.at[idx] = is_dup
            if not is_dup:
                seen.append(row_str)
        return mask

    @staticmethod
    def _char_similarity(a: str, b: str) -> float:
        """Simple Jaccard similarity on character bigrams."""
        if not a or not b:
            return 0.0
        bigrams_a = {a[i : i + 2] for i in range(len(a) - 1)}
        bigrams_b = {b[i : i + 2] for i in range(len(b) - 1)}
        if not bigrams_a or not bigrams_b:
            return 0.0
        return len(bigrams_a & bigrams_b) / len(bigrams_a | bigrams_b)
