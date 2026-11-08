"""Shared data-contract tests with no services or model dependencies."""

from dataclasses import FrozenInstanceError, asdict, fields, replace

import pytest

from java_docs_rag.common.models import (
    Document,
    DocumentChunk,
    EvaluationQuestion,
    RAGAnswer,
    SearchResult,
)


@pytest.fixture
def document() -> Document:
    return Document(
        document_id="java-list",
        source_url="https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/List.html",
        title="List",
        content="An ordered collection.",
        content_hash="example-sha256",
    )


@pytest.fixture
def chunk(document: Document) -> DocumentChunk:
    return DocumentChunk(
        chunk_id="java-list:0",
        document_id=document.document_id,
        text=document.content,
        source_url=document.source_url,
        title=document.title,
        section=None,
        position=0,
    )


def test_document_retains_source_fields(document: Document) -> None:
    assert document.document_id == "java-list"
    assert document.title == "List"
    assert document.content == "An ordered collection."
    assert document.content_hash == "example-sha256"
    assert document.updated_at is None
    assert document.metadata == {}


def test_document_accepts_timestamp_and_metadata(document: Document) -> None:
    updated = replace(document, updated_at="2026-11-08T10:00:00Z", metadata={"java_version": "21"})
    assert updated.updated_at == "2026-11-08T10:00:00Z"
    assert updated.metadata == {"java_version": "21"}
    assert document.updated_at is None


def test_chunk_retains_document_identity_and_position(
    document: Document, chunk: DocumentChunk
) -> None:
    assert chunk.document_id == document.document_id
    assert chunk.source_url == document.source_url
    assert chunk.title == document.title
    assert chunk.text == document.content
    assert chunk.position == 0
    assert chunk.section is None
    assert replace(chunk, section="Methods", position=1).section == "Methods"


@pytest.mark.parametrize("model_name", ["document", "chunk"])
def test_metadata_defaults_are_independent(
    model_name: str, document: Document, chunk: DocumentChunk
) -> None:
    original = document if model_name == "document" else chunk
    values = asdict(original)
    values.pop("metadata")
    other = type(original)(**values)
    original.metadata["version"] = "21"
    assert other.metadata == {}


def test_search_result_preserves_scores_and_chunk(chunk: DocumentChunk) -> None:
    result = SearchResult(chunk=chunk, similarity_score=0.85)
    assert result.chunk is chunk
    assert result.similarity_score == 0.85
    assert result.rerank_score is None
    reranked = replace(result, rerank_score=2.5)
    assert reranked.rerank_score == 2.5
    assert reranked.similarity_score == result.similarity_score


def test_grounded_answer_preserves_sources_and_retrieval(chunk: DocumentChunk) -> None:
    result = SearchResult(chunk, similarity_score=0.9)
    answer = RAGAnswer(
        question="What is a List?",
        answer="An ordered collection.",
        sources=[chunk],
        retrieved_chunks=[result],
        model="test-local-model",
        insufficient_context=False,
    )
    assert answer.question == "What is a List?"
    assert answer.answer == chunk.text
    assert answer.sources == [chunk]
    assert answer.retrieved_chunks == [result]
    assert answer.model == "test-local-model"
    assert answer.insufficient_context is False


def test_answer_can_express_insufficient_context() -> None:
    answer = RAGAnswer("Unknown?", "Insufficient context.", [], [], "test", True)
    assert answer.insufficient_context is True
    assert answer.sources == []
    assert answer.retrieved_chunks == []


@pytest.mark.parametrize("answerable", [True, False])
def test_evaluation_question_supports_answerability(answerable: bool) -> None:
    expected = "An ordered collection." if answerable else None
    relevant = ["java-list"] if answerable else []
    question = EvaluationQuestion("q1", "What is a List?", expected, relevant, "api", answerable)
    assert question.question_id == "q1"
    assert question.question == "What is a List?"
    assert question.expected_answer == expected
    assert question.relevant_document_ids == relevant
    assert question.category == "api"
    assert question.answerable is answerable


@pytest.mark.parametrize("model_name", ["document", "chunk", "result", "answer", "evaluation"])
def test_models_reject_field_reassignment(
    model_name: str, document: Document, chunk: DocumentChunk
) -> None:
    models = {
        "document": document,
        "chunk": chunk,
        "result": SearchResult(chunk, 0.5),
        "answer": RAGAnswer("q", "a", [], [], "test", False),
        "evaluation": EvaluationQuestion("q1", "q", None, [], "api", False),
    }
    model = models[model_name]
    name = fields(model)[0].name
    with pytest.raises(FrozenInstanceError):
        setattr(model, name, "changed")


def test_nested_answer_serializes_to_plain_data(chunk: DocumentChunk) -> None:
    answer = RAGAnswer("q", "a", [chunk], [SearchResult(chunk, 0.5)], "test", False)
    data = asdict(answer)
    assert data["sources"][0]["chunk_id"] == chunk.chunk_id
    assert data["retrieved_chunks"][0]["similarity_score"] == 0.5
    assert data["retrieved_chunks"][0]["rerank_score"] is None
    assert data["insufficient_context"] is False
