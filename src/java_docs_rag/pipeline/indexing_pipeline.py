"""Developer B: indexing orchestration boundary; no pipeline implementation."""

from collections.abc import Sequence
from dataclasses import dataclass

from java_docs_rag.common.interfaces import Chunker, Embedder, VectorStore
from java_docs_rag.common.models import Document


@dataclass
class IndexingPipeline:
    chunker: Chunker
    embedder: Embedder
    vectorstore: VectorStore

    def run(self, documents: Sequence[Document]) -> None:
        raise NotImplementedError("Indexing pipeline is not implemented")
