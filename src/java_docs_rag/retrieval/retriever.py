"""Developer A: vector retrieval through shared interfaces."""

from dataclasses import dataclass

from java_docs_rag.common.interfaces import Embedder, VectorStore
from java_docs_rag.common.models import SearchResult


@dataclass
class VectorRetriever:
    embedder: Embedder
    vectorstore: VectorStore

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        raise NotImplementedError("Retrieval is not implemented")
