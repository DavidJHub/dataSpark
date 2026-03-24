"""Duplicate detection and removal helpers.

This module provides exact and fuzzy deduplication utilities for tabular data.
The fuzzy mode uses a lightweight character-bigram Jaccard similarity.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import pandas as pd
from loguru import logger


class Deduplicator:
    """Detect and remove duplicate records using exact or fuzzy matching.

    Parameters
    ----------
    strategy:
        Duplicate strategy:

        - ``"exact"`` uses :meth:`pandas.DataFrame.duplicated` semantics.
        - ``"fuzzy"`` compares row text representations with a similarity
          threshold.
    similarity_threshold:
        Minimum similarity score (0-1) used in fuzzy mode.
    subset:
        Optional list of columns that define the record identity.
    """

    def __init__(
        self,
        strategy: Literal["exact", "fuzzy"] = "exact",
        similarity_threshold: float = 0.85,
        subset: list[str] | None = None,
    ) -> None:
        """Initialize deduplicator configuration."""
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
        self.subset = subset

    def find_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return rows identified as duplicates.

        In exact mode, the first occurrence is considered canonical and only
        subsequent duplicates are returned.
        """
        if self.strategy == "exact":
            mask = df.duplicated(subset=self.subset, keep="first")
        else:
            mask = self._fuzzy_duplicates(df)
        n = mask.sum()
        logger.info("Found {} duplicate rows (strategy={})", n, self.strategy)
        return df[mask]

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates and return the resulting dataframe."""
        if self.strategy == "exact":
            result = df.drop_duplicates(subset=self.subset, keep="first").reset_index(drop=True)
        else:
            mask = self._fuzzy_duplicates(df)
            result = df[~mask].reset_index(drop=True)
        removed = len(df) - len(result)
        logger.info("Removed {} duplicates — {} → {} rows", removed, len(df), len(result))
        return result

    def hash_rows(self, df: pd.DataFrame) -> pd.Series:
        """Compute SHA-256 hash digest for each row.

        This helper is useful for large-scale exact deduplication pipelines
        where deterministic row fingerprints are required.
        """
        cols = self.subset or df.columns.tolist()
        return df[cols].apply(
            lambda row: hashlib.sha256(str(row.values).encode()).hexdigest(), axis=1
        )

    def _fuzzy_duplicates(self, df: pd.DataFrame) -> pd.Series:
        """Build duplicate mask using pairwise fuzzy text comparison.

        The algorithm performs greedy matching against previously kept rows.
        Complexity is approximately O(n²) in the number of rows.
        """
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
        """Compute Jaccard similarity using character bigrams.

        Parameters
        ----------
        a, b:
            Input strings.

        Returns
        -------
        float
            Similarity score between 0 and 1.
        """
        if not a or not b:
            return 0.0
        bigrams_a = {a[i : i + 2] for i in range(len(a) - 1)}
        bigrams_b = {b[i : i + 2] for i in range(len(b) - 1)}
        if not bigrams_a or not bigrams_b:
            return 0.0
        return len(bigrams_a & bigrams_b) / len(bigrams_a | bigrams_b)
