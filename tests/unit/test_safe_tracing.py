"""Count-only tracing, business error privacy and fail-open telemetry without a server."""

from contextlib import contextmanager
from dataclasses import replace

import pytest
from test_rag_pipeline import Embedder, Provider, Store

from java_docs_rag.common.config import ObservabilityConfig
from java_docs_rag.observability.langfuse_client import LangfuseClient
from java_docs_rag.observability.safe_tracing import SafeTelemetry
from java_docs_rag.pipeline.rag_pipeline import RAGPipeline
from java_docs_rag.reranking.cross_encoder import CrossEncoderReranker
from java_docs_rag.retrieval.retriever import SemanticRetriever


class RecordingTracer:
    enabled = True

    def __init__(self, failure=None):
        self.events = []
        self.failure = failure
        self.flushes = 0
        self.client = self

    def get_current_trace_id(self):
        return "a" * 32

    @contextmanager
    def span(self, name, metadata=None, input=None):
        if self.failure == "start":
            raise RuntimeError("credential-SENSITIVE")
        event = {
            "name": name,
            "metadata": metadata,
            "input": input,
            "output": None,
            "exception_received": False,
        }
        self.events.append(event)
        failure = self.failure

        class Handle:
            def update(self, **fields):
                if failure == "update":
                    raise RuntimeError("credential-SENSITIVE")
                event.update(fields)

        try:
            yield Handle()
        except BaseException:
            event["exception_received"] = True
            raise
        finally:
            if self.failure == "close":
                raise RuntimeError("credential-SENSITIVE")

    start_as_current_span = span

    def flush(self):
        self.flushes += 1
        if self.failure == "flush":
            raise RuntimeError("credential-SENSITIVE")


def test_privacy_allowlist_and_no_exception_forwarding() -> None:
    backend = RecordingTracer()
    tracer = SafeTelemetry(backend)
    error = ValueError("question-SENSITIVE prompt-SENSITIVE key-SENSITIVE")
    with pytest.raises(ValueError) as caught:
        with tracer.span(
            "stage", {"documents": 2, "question": "question-SENSITIVE"}, input="document-SENSITIVE"
        ) as span:
            span.update(
                output={
                    "source_count": 1,
                    "answer": "answer-SENSITIVE",
                    "metadata": {"secret": "key-SENSITIVE"},
                    "selected_chars": float("nan"),
                }
            )
            raise error
    assert caught.value is error
    event = backend.events[0]
    assert event["input"] is None and not event["exception_received"]
    assert event["output"]["outcome"] == "error"
    assert event["output"]["documents"] == 2
    assert event["output"]["duration_ms"] >= 0
    assert "SENSITIVE" not in repr(backend.events)
    assert tracer.summary()["receipt_verified"] is False
    assert tracer.trace_ids == ["a" * 32]


@pytest.mark.parametrize("failure", ["start", "update", "close", "flush"])
def test_telemetry_lifecycle_failures_do_not_change_business_result(failure, caplog) -> None:
    tracer = SafeTelemetry(RecordingTracer(failure))
    with tracer.span("operation"):
        result = 42
    tracer.flush()
    assert result == 42 and tracer.errors > 0
    assert "SENSITIVE" not in caplog.text


@pytest.mark.parametrize("failure", ["start", "update", "close", "flush"])
def test_existing_sdk_adapter_errors_are_sanitized_and_business_errors_survive(failure, caplog):
    sdk = RecordingTracer(failure)
    tracer = SafeTelemetry(LangfuseClient(ObservabilityConfig(enabled=True), client=sdk))
    with pytest.raises(ValueError, match="business-SENSITIVE"):
        with tracer.span("operation"):
            raise ValueError("business-SENSITIVE")
    tracer.flush()
    assert "SENSITIVE" not in caplog.text
    assert all(not event["exception_received"] for event in sdk.events)


@pytest.mark.parametrize("empty", [False, True])
def test_rag_stage_counts_outcomes_and_privacy(candidates, empty) -> None:
    backend = RecordingTracer()
    tracer = SafeTelemetry(backend)
    candidates = [
        replace(item, chunk=replace(item.chunk, text="document-SENSITIVE")) for item in candidates
    ]
    provider = Provider("answer-SENSITIVE")
    pipeline = RAGPipeline(
        SemanticRetriever(Embedder(), Store([] if empty else candidates)),
        CrossEncoderReranker("fake", lambda pairs: [1.0] * len(pairs)),
        provider,
        tracer=tracer,
    )
    answer = pipeline.run("question-SENSITIVE")
    assert answer.insufficient_context is empty
    assert [event["name"] for event in backend.events] == [
        "rag.run",
        "rag.retrieve",
        "rag.similarity_filter",
        "rag.rerank",
        "rag.evidence_selection",
        "rag.generate",
    ]
    assert all(event["output"]["duration_ms"] >= 0 for event in backend.events)
    generated = backend.events[-1]["output"]
    assert generated["provider_called"] is (not empty)
    assert generated["outcome"] == ("abstained" if empty else "ok")
    assert "SENSITIVE" not in repr(backend.events)
    assert backend.flushes == 1


def test_rag_failure_outcome_is_count_only_and_original_error_propagates(candidates):
    backend = RecordingTracer()

    class Broken:
        def retrieve(self, query, top_k):
            raise RuntimeError("question-SENSITIVE")

    pipeline = RAGPipeline(Broken(), None, Provider(), tracer=SafeTelemetry(backend))
    with pytest.raises(RuntimeError, match="question-SENSITIVE"):
        pipeline.run("question-SENSITIVE")
    assert [event["output"]["outcome"] for event in backend.events] == ["error", "error"]
    assert "SENSITIVE" not in repr(backend.events)
    assert backend.flushes == 1
