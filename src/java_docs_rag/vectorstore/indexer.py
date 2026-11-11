"""Developer B: batching and indexing coordinator (chunks → embeddings → VectorStore)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from java_docs_rag.common.exceptions import EmbeddingError
from java_docs_rag.common.interfaces import Embedder, VectorStore
from java_docs_rag.common.models import DocumentChunk


def embedding_text(chunk: DocumentChunk, include_context: bool = True) -> str:
    """Text sent to the embedder. With context, the page title and section path are
    prepended so that e.g. a ``size()`` method chunk still mentions ``List``.
    The stored chunk text is not changed."""
    if not include_context:
        return chunk.text
    header = chunk.title
    if chunk.section and chunk.section != chunk.title:
        header = f"{header} — {chunk.section}" if header else chunk.section
    return f"{header}\n{chunk.text}" if header else chunk.text


@dataclass
class IndexingStats:
    chunks: int = 0
    batches: int = 0
    embed_seconds: float = 0.0
    upsert_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return self.embed_seconds + self.upsert_seconds

    def add(self, other: IndexingStats) -> None:
        self.chunks += other.chunks
        self.batches += other.batches
        self.embed_seconds += other.embed_seconds
        self.upsert_seconds += other.upsert_seconds


@dataclass
class DocumentIndexer:
    embedder: Embedder
    vectorstore: VectorStore
    batch_size: int = 64
    include_context: bool = True
    stats: IndexingStats = field(default_factory=IndexingStats)

    def index(self, chunks: Sequence[DocumentChunk]) -> IndexingStats:
        run = IndexingStats()
        for start in range(0, len(chunks), self.batch_size):
            batch = list(chunks[start : start + self.batch_size])
            texts = [embedding_text(chunk, self.include_context) for chunk in batch]
            began = time.perf_counter()
            vectors = self.embedder.embed_documents(texts)
            embedded = time.perf_counter()
            if len(vectors) != len(batch):
                raise EmbeddingError(f"Embedder returned {len(vectors)} vectors for {len(batch)}")
            dimension = self.embedder.dimension
            if any(len(vector) != dimension for vector in vectors):
                raise EmbeddingError(f"Vector dimension differs from embedder ({dimension})")
            self.vectorstore.upsert(batch, vectors)
            run.embed_seconds += embedded - began
            run.upsert_seconds += time.perf_counter() - embedded
            run.chunks += len(batch)
            run.batches += 1
        self.stats.add(run)
        return run
