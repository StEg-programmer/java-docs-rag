"""Developer B: paragraph chunking — whole paragraphs (and code blocks) packed up to
``chunk_size``; a single paragraph longer than the limit is split at whitespace."""

from java_docs_rag.preprocessing.chunking.base import BaseChunker, Span, find_blocks


class ParagraphChunker(BaseChunker):
    strategy = "paragraph"
    uses_overlap = False

    def spans(self, text: str) -> list[Span]:
        return self.pack_blocks(text, find_blocks(text))
