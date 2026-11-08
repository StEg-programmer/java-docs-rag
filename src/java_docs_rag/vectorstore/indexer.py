"""Developer B: future batching and indexing coordinator."""

from collections.abc import Sequence
from dataclasses import dataclass

from java_docs_rag.common.interfaces import Embedder, VectorStore
from java_docs_rag.common.models import DocumentChunk


@dataclass
class DocumentIndexer:
    embedder: Embedder
    vectorstore: VectorStore

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        raise NotImplementedError("Chunk indexing is not implemented")
