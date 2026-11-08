"""Developer B: paragraph chunking strategy placeholder."""

from java_docs_rag.common.models import Document, DocumentChunk
from java_docs_rag.preprocessing.chunking.base import BaseChunker


class ParagraphChunker(BaseChunker):
    def split(self, document: Document) -> list[DocumentChunk]:
        raise NotImplementedError("ParagraphChunker is not implemented")
