"""Developer B: fixed-size chunking (no overlap, breaks at whitespace when possible)."""

from java_docs_rag.preprocessing.chunking.base import BaseChunker, Span, window_spans


class FixedChunker(BaseChunker):
    strategy = "fixed"
    uses_overlap = False

    def spans(self, text: str) -> list[Span]:
        return window_spans(text, 0, len(text), self.chunk_size, 0)
