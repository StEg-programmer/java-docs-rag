"""Synchronous structural contracts shared by both developers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from java_docs_rag.common.models import Document, DocumentChunk, SearchResult


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class Chunker(Protocol):
    def split(self, document: Document) -> list[DocumentChunk]: ...


class VectorStore(Protocol):
    def upsert(
        self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]]
    ) -> None: ...

    def search(
        self, vector: Sequence[float], top_k: int, filters: dict[str, str] | None = None
    ) -> list[SearchResult]: ...

    def delete_document(self, document_id: str) -> None: ...


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[SearchResult]: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, results: Sequence[SearchResult], top_n: int
    ) -> list[SearchResult]: ...


class LLMProvider(Protocol):
    @property
    def model_id(self) -> str: ...

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...
