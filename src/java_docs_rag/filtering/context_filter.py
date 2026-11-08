"""Developer A: select sufficient evidence within a context budget."""

from collections.abc import Sequence

from java_docs_rag.common.models import SearchResult


class ContextFilter:
    def filter(self, results: Sequence[SearchResult], max_chars: int) -> list[SearchResult]:
        raise NotImplementedError("Context filtering is not implemented")
