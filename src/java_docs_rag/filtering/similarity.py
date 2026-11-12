"""Developer A: threshold filtering using the configured score semantics."""

from collections.abc import Sequence
from math import isfinite

from java_docs_rag.common.models import SearchResult


class SimilarityFilter:
    def filter(
        self, results: Sequence[SearchResult], threshold: float | None
    ) -> list[SearchResult]:
        """Keep scores >= threshold in input order; None disables the cutoff."""
        if threshold is None:
            return list(results)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a finite number or None")
        if not isfinite(threshold):
            raise ValueError("threshold must be finite")
        return [result for result in results if result.similarity_score >= threshold]
