"""Shared data contracts. Frozen fields do not deeply freeze lists or dictionaries."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    document_id: str
    source_url: str
    title: str
    content: str
    content_hash: str
    updated_at: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    source_url: str
    title: str
    section: str | None
    position: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    similarity_score: float
    rerank_score: float | None = None


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    answer: str
    sources: list[DocumentChunk]
    retrieved_chunks: list[SearchResult]
    model: str
    insufficient_context: bool


@dataclass(frozen=True)
class EvaluationQuestion:
    question_id: str
    question: str
    expected_answer: str | None
    relevant_document_ids: list[str]
    category: str
    answerable: bool
