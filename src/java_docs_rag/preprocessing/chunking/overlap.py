"""Developer B: sliding-window chunking; neighbours share ``config.overlap`` characters."""

from java_docs_rag.preprocessing.chunking.base import BaseChunker, Span, window_spans


class OverlapChunker(BaseChunker):
    strategy = "overlap"
    uses_overlap = True

    def spans(self, text: str) -> list[Span]:
        return window_spans(text, 0, len(text), self.chunk_size, self.overlap)
