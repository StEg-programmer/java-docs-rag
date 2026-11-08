"""Developer B: section chunking strategy placeholder."""

from java_docs_rag.common.models import Document, DocumentChunk
from java_docs_rag.preprocessing.chunking.base import BaseChunker


class SectionChunker(BaseChunker):
    def split(self, document: Document) -> list[DocumentChunk]:
        raise NotImplementedError("SectionChunker is not implemented")
