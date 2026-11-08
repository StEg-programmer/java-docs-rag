"""Developer A: remove duplicate retrieved chunks."""

from collections.abc import Sequence

from java_docs_rag.common.models import SearchResult


class DeduplicationFilter:
    def filter(self, results: Sequence[SearchResult]) -> list[SearchResult]:
        raise NotImplementedError("Deduplication is not implemented")
