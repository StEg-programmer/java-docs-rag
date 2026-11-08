"""Developer B: future Qdrant adapter; constructing it has no network side effects."""

from collections.abc import Sequence
from dataclasses import dataclass

from java_docs_rag.common.config import VectorStoreConfig
from java_docs_rag.common.models import DocumentChunk, SearchResult


@dataclass
class QdrantVectorStore:
    config: VectorStoreConfig

    def upsert(self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]]) -> None:
        raise NotImplementedError("Vector upsert is not implemented")

    def search(
        self, vector: Sequence[float], top_k: int, filters: dict[str, str] | None = None
    ) -> list[SearchResult]:
        raise NotImplementedError("Vector search is not implemented")

    def delete_document(self, document_id: str) -> None:
        raise NotImplementedError("Document deletion is not implemented")
