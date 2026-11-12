"""Developer A: select sufficient evidence within a context budget."""

from collections.abc import Sequence

from java_docs_rag.common.models import SearchResult


class ContextFilter:
    def filter(self, results: Sequence[SearchResult], max_chars: int) -> list[SearchResult]:
        """Select whole chunks with sum(len(chunk.text)) <= max_chars.

        Count Python Unicode characters, including whitespace, only in chunk text;
        prompt instructions, source labels and serialization overhead are excluded.
        Skip chunks that do not fit and continue. Zero is a valid budget.
        """
        if isinstance(max_chars, bool) or not isinstance(max_chars, int):
            raise TypeError("max_chars must be an integer")
        if max_chars < 0:
            raise ValueError("max_chars must be non-negative")
        selected = []
        remaining = max_chars
        for result in results:
            cost = len(result.chunk.text)
            if cost <= remaining:
                selected.append(result)
                remaining -= cost
        return selected
