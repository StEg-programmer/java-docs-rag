"""Semantic retrieval behavior using only in-memory protocol implementations."""

from collections.abc import Sequence
from dataclasses import replace

import pytest

from java_docs_rag.common.exceptions import EmbeddingError, VectorStoreError
from java_docs_rag.common.interfaces import Retriever
from java_docs_rag.common.models import DocumentChunk, SearchResult
from java_docs_rag.retrieval.retriever import SemanticRetriever, VectorRetriever


class FakeEmbedder:
    model_id = "fake"
    dimension = 3

    def __init__(self) -> None:
        self.vector = [0.1, 0.2, 0.3]
        self.queries: list[str] = []
        self.error: EmbeddingError | None = None

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        if self.error is not None:
            raise self.error
        return self.vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("Retrieval must use embed_query")


class FakeVectorStore:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.searches: list[tuple[list[float], int, dict[str, str] | None]] = []
        self.error: VectorStoreError | None = None

    def search(
        self, vector: Sequence[float], top_k: int, filters: dict[str, str] | None = None
    ) -> list[SearchResult]:
        self.searches.append((list(vector), top_k, filters))
        if self.error is not None:
            raise self.error
        return self.results[:top_k]

    def upsert(self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]]) -> None:
        raise AssertionError("Retrieval must not write to the store")

    def delete_document(self, document_id: str) -> None:
        raise AssertionError("Retrieval must not delete documents")


@pytest.fixture
def results() -> list[SearchResult]:
    chunk = DocumentChunk(
        chunk_id="list:0",
        document_id="list",
        text="An ordered collection.",
        source_url="https://example.test/java/util/List.html",
        title="List",
        section=None,
        position=0,
        metadata={"java_version": "21"},
    )
    return [
        SearchResult(chunk, similarity_score=0.4, rerank_score=0.2),
        SearchResult(replace(chunk, chunk_id="list:1", position=1), similarity_score=0.9),
    ]


@pytest.mark.parametrize("retriever_type", [SemanticRetriever, VectorRetriever])
@pytest.mark.parametrize("top_k", [1, 2, 10])
def test_retrieve_embeds_query_and_preserves_store_results(
    retriever_type: type[SemanticRetriever], top_k: int, results: list[SearchResult]
) -> None:
    embedder = FakeEmbedder()
    store = FakeVectorStore(results)
    retriever: Retriever = retriever_type(embedder=embedder, vectorstore=store)
    query = "  What is a Java List?\n"

    actual = retriever.retrieve(query=query, top_k=top_k)

    assert embedder.queries == [query]
    assert store.searches == [(embedder.vector, top_k, None)]
    assert actual == results[:top_k]
    assert all(result is expected for result, expected in zip(actual, results, strict=False))
    assert embedder.vector == [0.1, 0.2, 0.3]
    assert results[0].chunk.metadata == {"java_version": "21"}


def test_retrieve_returns_empty_search_results() -> None:
    embedder = FakeEmbedder()
    store = FakeVectorStore([])

    assert SemanticRetriever(embedder, store).retrieve("Unknown Java API?", 5) == []
    assert embedder.queries == ["Unknown Java API?"]
    assert store.searches == [(embedder.vector, 5, None)]


@pytest.mark.parametrize("query", ["", " ", "\t\r\n", "\u2003"])
def test_retrieve_rejects_blank_questions_before_adapter_calls(query: str) -> None:
    embedder = FakeEmbedder()
    store = FakeVectorStore([])

    with pytest.raises(ValueError, match="query"):
        SemanticRetriever(embedder, store).retrieve(query, 1)

    assert embedder.queries == []
    assert store.searches == []


@pytest.mark.parametrize("query", [None, 42, [], b"List?"])
def test_retrieve_rejects_non_string_questions(query: object) -> None:
    embedder = FakeEmbedder()
    store = FakeVectorStore([])

    with pytest.raises(TypeError, match="query"):
        SemanticRetriever(embedder, store).retrieve(query, 1)  # type: ignore[arg-type]

    assert embedder.queries == []
    assert store.searches == []


@pytest.mark.parametrize(
    ("top_k", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
        (1.0, TypeError),
        ("5", TypeError),
        (None, TypeError),
        (True, TypeError),
        (False, TypeError),
    ],
)
def test_retrieve_rejects_invalid_top_k_before_adapter_calls(
    top_k: object, error_type: type[Exception]
) -> None:
    embedder = FakeEmbedder()
    store = FakeVectorStore([])

    with pytest.raises(error_type, match="top_k"):
        SemanticRetriever(embedder, store).retrieve("List?", top_k)  # type: ignore[arg-type]

    assert embedder.queries == []
    assert store.searches == []


def test_embedding_failure_propagates_without_searching() -> None:
    embedder = FakeEmbedder()
    embedder.error = EmbeddingError("Embedding unavailable")
    store = FakeVectorStore([])

    with pytest.raises(EmbeddingError) as caught:
        SemanticRetriever(embedder, store).retrieve("List?", 1)

    assert caught.value is embedder.error
    assert store.searches == []


def test_search_failure_propagates() -> None:
    embedder = FakeEmbedder()
    store = FakeVectorStore([])
    store.error = VectorStoreError("Store unavailable")

    with pytest.raises(VectorStoreError) as caught:
        SemanticRetriever(embedder, store).retrieve("List?", 1)

    assert caught.value is store.error
    assert embedder.queries == ["List?"]
