"""Developer A: remove duplicate retrieved chunks."""

from collections.abc import Sequence

from java_docs_rag.common.models import SearchResult


class DeduplicationFilter:
    def filter(self, results: Sequence[SearchResult]) -> list[SearchResult]:
        """Keep the first result for each chunk_id, retaining ranking order."""
        seen: set[str] = set()
        unique = []
        for result in results:
            if result.chunk.chunk_id not in seen:
                unique.append(result)
                seen.add(result.chunk.chunk_id)
        return unique
