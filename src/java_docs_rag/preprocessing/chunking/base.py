"""Developer B: common base for configurable chunking strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from java_docs_rag.common.config import ChunkingConfig
from java_docs_rag.common.models import Document, DocumentChunk


@dataclass
class BaseChunker(ABC):
    config: ChunkingConfig

    @abstractmethod
    def split(self, document: Document) -> list[DocumentChunk]:
        """Split a document while retaining source identity and order."""
        raise NotImplementedError
