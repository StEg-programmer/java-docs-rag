"""Offline A/B round trip: existing HTML → real indexing → memory → real answer pipeline."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from java_docs_rag.common.config import ChunkingConfig, EmbeddingConfig, RetrievalConfig
from java_docs_rag.embeddings.factory import create_embedder
from java_docs_rag.generation.generator import AnswerGenerator
from java_docs_rag.generation.prompt_builder import ABSTENTION_MARKER
from java_docs_rag.grabber.identity import content_hash
from java_docs_rag.pipeline.indexing_pipeline import DocumentPreparer, IndexingPipeline
from java_docs_rag.pipeline.rag_pipeline import RAGPipeline
from java_docs_rag.preprocessing.chunking.factory import create_chunker
from java_docs_rag.reranking.cross_encoder import CrossEncoderReranker
from java_docs_rag.reranking.sentence_transformer import SentenceTransformerScorer
from java_docs_rag.retrieval.retriever import SemanticRetriever
from java_docs_rag.vectorstore.memory import InMemoryVectorStore

FIXTURE = Path(__file__).parents[1] / "fixtures/javadoc/java.base/java/util/List.html"
SOURCE_URL = "https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/List.html"
DOCUMENT_ID = "javase21:java.base/java/util/List"


class DeterministicEmbeddingModel:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return [
            ([0.0, 1.0] if "weather" in text.lower() else [1.0, 0.0]) + [0.0] * 382
            for text in texts
        ]


class DeterministicScoringModel:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append(list(pairs))
        return [float("number of elements" in text.lower()) for _, text in pairs]


class EvidenceProvider:
    model_id = "deterministic-test-provider"

    def __init__(self) -> None:
        self.calls = []

    def generate(self, system, user):
        self.calls.append((system, user))
        evidence = json.loads(user.split("Evidence JSON:\n", 1)[1])
        if "weather" in user.lower():
            return ABSTENTION_MARKER
        assert "number of elements" in evidence[0]["text"].lower()
        return "List.size() returns the number of elements in this list [S1]."


def test_local_javadoc_indexes_and_answers_with_expected_source() -> None:
    document = DocumentPreparer().prepare(FIXTURE.read_bytes(), SOURCE_URL)
    assert document is not None and document.document_id == DOCUMENT_ID
    model = DeterministicEmbeddingModel()
    embedder = create_embedder(EmbeddingConfig("bge-small"))
    embedder._loader = lambda *_: model  # Replace only model inference, not the B adapter.
    chunker = create_chunker(ChunkingConfig("paragraph", 500, 150))
    store = InMemoryVectorStore(dimension=embedder.dimension)
    report = IndexingPipeline(chunker, embedder, store).run([document])
    expected_chunks = {chunk.chunk_id: chunk for chunk in chunker.split(document)}
    assert report.documents_indexed == 1
    assert report.chunks == len(store) == len(expected_chunks) > 1
    assert store.dimension == embedder.dimension == 384
    assert embedder.model_id == "BAAI/bge-small-en-v1.5"
    assert model.calls and document.title in model.calls[0][0]

    scoring_model, provider = DeterministicScoringModel(), EvidenceProvider()
    scorer = SentenceTransformerScorer("fake-local", model_loader=lambda *_: scoring_model)
    pipeline = RAGPipeline(
        SemanticRetriever(embedder, store),
        CrossEncoderReranker(scorer.model_id, scorer),
        provider,
        RetrievalConfig(top_k=20, rerank_top_n=3, min_similarity=0.5),
    )
    answer = pipeline.run("  What does List.size() return?  ")
    assert answer.question == "What does List.size() return?"
    assert not answer.insufficient_context and "[S1]" in answer.answer
    assert answer.sources
    for source in answer.sources:
        assert source == expected_chunks[source.chunk_id]
        assert source.document_id == DOCUMENT_ID and source.source_url == SOURCE_URL
        assert source.metadata["content_hash"] == document.content_hash
        assert source.chunk_id.startswith(DOCUMENT_ID + "#paragraph-s500-o0#")
    assert all(hit.similarity_score == pytest.approx(1) for hit in answer.retrieved_chunks)
    assert all(hit.rerank_score is None for hit in answer.retrieved_chunks)
    assert model.calls[-1] == [embedder.query_prefix + answer.question]
    assert scoring_model.calls[0] == [
        (answer.question, hit.chunk.text) for hit in answer.retrieved_chunks
    ]
    evidence = json.loads(provider.calls[0][1].split("Evidence JSON:\n", 1)[1])
    assert [row["chunk_id"] for row in evidence] == [source.chunk_id for source in answer.sources]

    outside = pipeline.run("What is tomorrow's weather?")
    assert outside.insufficient_context and outside.sources == []
    assert outside.retrieved_chunks  # Raw retrieval retained despite filtering all evidence.
    assert len(provider.calls) == len(scoring_model.calls) == 1

    # The model-declared path is also covered with irrelevant evidence that survives cutoff.
    model_refusal = AnswerGenerator(provider).generate(outside.question, outside.retrieved_chunks)
    assert model_refusal.insufficient_context and model_refusal.sources

    # Actual incremental B pipeline removes old tail chunks of an edited document.
    shorter = replace(
        document,
        content="List.size returns a count.",
        content_hash=content_hash("List.size returns a count."),
    )
    replaced = IndexingPipeline(chunker, embedder, store).run(
        [shorter], previous_hashes=report.hashes
    )
    assert replaced.chunks == len(store) == 1
