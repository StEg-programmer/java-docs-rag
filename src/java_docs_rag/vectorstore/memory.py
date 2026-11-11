"""Developer B: exact in-memory VectorStore (cosine) for tests and small experiments.

Results are comparable to Qdrant's Cosine distance; nothing is persisted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from java_docs_rag.common.exceptions import VectorStoreError
from java_docs_rag.common.models import DocumentChunk, SearchResult
from java_docs_rag.vectorstore.qdrant_store import CHUNK_FIELDS


def _unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else [0.0 for _ in vector]


def _matches(chunk: DocumentChunk, filters: dict[str, str] | None) -> bool:
    for key, value in (filters or {}).items():
        if key in CHUNK_FIELDS:
            actual = getattr(chunk, key)
        else:
            actual = chunk.metadata.get(key.removeprefix("metadata."))
        if actual is None or str(actual) != value:
            return False
    return True


@dataclass
class InMemoryVectorStore:
    dimension: int | None = None
    _points: dict[str, tuple[DocumentChunk, list[float]]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self._points)

    def upsert(self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for chunk, vector in zip(chunks, vectors, strict=True):
            if self.dimension is None:
                self.dimension = len(vector)
            if len(vector) != self.dimension:
                raise VectorStoreError(f"Expected {self.dimension}-d vector, got {len(vector)}")
            self._points[chunk.chunk_id] = (chunk, _unit(vector))

    def search(
        self, vector: Sequence[float], top_k: int, filters: dict[str, str] | None = None
    ) -> list[SearchResult]:
        if self.dimension is not None and len(vector) != self.dimension:
            raise VectorStoreError(f"Expected {self.dimension}-d query, got {len(vector)}")
        query = _unit(vector)
        scored = [
            SearchResult(chunk, sum(a * b for a, b in zip(query, stored, strict=True)))
            for chunk, stored in self._points.values()
            if _matches(chunk, filters)
        ]
        scored.sort(key=lambda r: (-r.similarity_score, r.chunk.chunk_id))
        return scored[: max(top_k, 0)]

    def delete_document(self, document_id: str) -> None:
        for chunk_id in [k for k, (c, _) in self._points.items() if c.document_id == document_id]:
            del self._points[chunk_id]
