"""Exercise real retrieval-to-generation orchestration with fake external adapters."""

import json
from dataclasses import replace

import pytest

from java_docs_rag.common.config import RetrievalConfig
from java_docs_rag.common.exceptions import EmbeddingError
from java_docs_rag.generation.prompt_builder import ABSTENTION_MARKER
from java_docs_rag.pipeline.rag_pipeline import RAGPipeline
from java_docs_rag.reranking.cross_encoder import CrossEncoderReranker
from java_docs_rag.retrieval.retriever import SemanticRetriever


class Embedder:
    def __init__(self):
        self.queries = []

    def embed_query(self, query):
        self.queries.append(query)
        return [0.5, 0.5]


class Store:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, vector, top_k, filters=None):
        self.calls.append((vector, top_k))
        return self.results[:top_k]


class Provider:
    model_id = "fake"

    def __init__(self, response="List [S1]."):
        self.calls = []
        self.response = response

    def generate(self, system, user):
        self.calls.append((system, user))
        return self.response


def test_real_pipeline_order_raw_snapshot_and_normalized_query(candidates) -> None:
    # Low-similarity first candidate would consume the sole rerank slot if not removed.
    low, first, last = candidates[2], candidates[0], candidates[1]
    duplicate = replace(first, rerank_score=42)
    raw = [low, last, first, duplicate]
    embedder, store, provider = Embedder(), Store(raw), Provider()
    scorer_calls = []

    def score(pairs):
        scorer_calls.append(pairs)
        return [0.3, 0.9, 0.8]

    pipeline = RAGPipeline(
        SemanticRetriever(embedder, store),
        CrossEncoderReranker("fake", score),
        provider,
        config=RetrievalConfig(top_k=9, rerank_top_n=3, min_similarity=0.5),
        max_context_chars=4,
    )
    answer = pipeline.run(" \nList?\t ")

    assert embedder.queries == ["List?"]
    assert store.calls == [([0.5, 0.5], 9)]
    assert scorer_calls == [[("List?", item.chunk.text) for item in [last, first, duplicate]]]
    assert answer.question == "List?"
    assert answer.retrieved_chunks == raw
    assert all(a is b for a, b in zip(answer.retrieved_chunks, raw, strict=True))
    assert answer.sources == [first.chunk]
    assert not answer.insufficient_context
    context = json.loads(provider.calls[0][1].split("Evidence JSON:\n", 1)[1])
    assert [entry["chunk_id"] for entry in context] == [first.chunk.chunk_id]
    assert raw == [low, last, first, duplicate]
    assert first.rerank_score is None


def test_cutoff_prevents_low_scores_from_consuming_rerank_slots(candidates) -> None:
    calls = []

    def score(pairs):
        calls.append(pairs)
        return [1.0, 2.0]

    provider = Provider()
    pipeline = RAGPipeline(
        SemanticRetriever(Embedder(), Store([candidates[2], *candidates[:2]])),
        CrossEncoderReranker("fake", score),
        provider,
        RetrievalConfig(top_k=3, rerank_top_n=1, min_similarity=0.5),
    )
    answer = pipeline.run("List?")
    assert len(calls[0]) == 2
    assert answer.sources == [candidates[1].chunk]


@pytest.mark.parametrize("reason", ["empty", "cutoff", "budget"])
def test_no_evidence_abstains_without_calling_provider(candidates, reason) -> None:
    provider = Provider()
    raw = [] if reason == "empty" else candidates
    scorer_calls = []

    def score(pairs):
        scorer_calls.append(pairs)
        return [1.0] * len(pairs)

    pipeline = RAGPipeline(
        SemanticRetriever(Embedder(), Store(raw)),
        CrossEncoderReranker("fake", score),
        provider,
        RetrievalConfig(min_similarity=1.0 if reason == "cutoff" else None),
        max_context_chars=0 if reason == "budget" else 8000,
    )
    answer = pipeline.run("Unknown?")
    assert answer.insufficient_context
    assert answer.sources == []
    assert answer.retrieved_chunks == raw
    assert provider.calls == []
    if reason != "budget":
        assert scorer_calls == []


def test_model_declared_abstention_preserves_supplied_sources(candidates) -> None:
    pipeline = RAGPipeline(
        SemanticRetriever(Embedder(), Store(candidates)),
        CrossEncoderReranker("fake", lambda pairs: [1.0] * len(pairs)),
        Provider(ABSTENTION_MARKER),
    )
    answer = pipeline.run("Question outside Java?")
    assert answer.insufficient_context
    assert answer.sources == [result.chunk for result in candidates]
    assert answer.retrieved_chunks == candidates


def test_adapter_failure_propagates_without_provider_call(candidates) -> None:
    class BrokenEmbedder:
        def embed_query(self, query):
            raise EmbeddingError("offline")

    provider = Provider()
    pipeline = RAGPipeline(
        SemanticRetriever(BrokenEmbedder(), Store(candidates)),
        CrossEncoderReranker("fake", lambda pairs: []),
        provider,
    )
    with pytest.raises(EmbeddingError, match="offline"):
        pipeline.run("List?")
    assert provider.calls == []
