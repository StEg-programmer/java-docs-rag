"""Developer B: local raw-document repository contract."""

from pathlib import Path


class DocumentRepository:
    def save(self, document_id: str, content: bytes) -> Path:
        raise NotImplementedError("Raw document persistence is not implemented")

    def load(self, document_id: str) -> bytes:
        raise NotImplementedError("Raw document loading is not implemented")
