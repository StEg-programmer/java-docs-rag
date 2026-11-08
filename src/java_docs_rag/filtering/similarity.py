"""Developer A: threshold filtering using the configured score semantics."""

from collections.abc import Sequence

from java_docs_rag.common.models import SearchResult


class SimilarityFilter:
    def filter(self, results: Sequence[SearchResult], threshold: float) -> list[SearchResult]:
        raise NotImplementedError("Similarity filtering is not implemented")
