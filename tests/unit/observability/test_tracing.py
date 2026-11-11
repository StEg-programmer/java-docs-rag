"""Developer B: tracing adapter with fake Langfuse SDKs (v3 and v2 styles)."""

from contextlib import contextmanager

import pytest

from java_docs_rag.common.config import ObservabilityConfig
from java_docs_rag.common.exceptions import ConfigurationError
from java_docs_rag.observability.langfuse_client import (
    LangfuseClient,
    NoOpTracer,
    create_tracer,
)

ON = ObservabilityConfig(enabled=True)


class FakeV3:
    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.flushed = 0

    @contextmanager
    def start_as_current_span(self, name, input=None, metadata=None):
        span = FakeSpan(name, self.events)
        self.events.append(("start", name, metadata))
        try:
            yield span
        finally:
            self.events.append(("end", name))

    def create_event(self, name, metadata):
        self.events.append(("event", name, metadata))

    def flush(self):
        self.flushed += 1


class FakeSpan:
    def __init__(self, name, events, parent=None):
        self.name, self.events, self.parent = name, events, parent

    def update(self, **fields):
        self.events.append(("update", self.name, fields))

    # v2 API
    def span(self, name, input=None, metadata=None):
        self.events.append(("start", name, metadata, self.name))
        return FakeSpan(name, self.events, self)

    def end(self):
        self.events.append(("end", self.name))


class FakeV2:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def trace(self, name, input=None, metadata=None):
        self.events.append(("start", name, metadata, None))
        return FakeSpan(name, self.events)

    def event(self, name, metadata):
        self.events.append(("event", name, metadata))

    def flush(self):
        pass


def test_disabled_config_gives_noop_tracer() -> None:
    tracer = create_tracer(ObservabilityConfig())
    assert isinstance(tracer, NoOpTracer) and not tracer.enabled
    with tracer.span("x", {"a": 1}) as span:
        span.update(output={"ok": True})
    tracer.record("e", {})
    tracer.flush()


def test_enabled_tracer_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tracer = create_tracer(ON)
    assert isinstance(tracer, LangfuseClient)
    with pytest.raises(ConfigurationError, match="LANGFUSE_PUBLIC_KEY"):
        with tracer.span("x"):
            pass


def test_from_env_reads_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    client = LangfuseClient.from_env(ON)
    assert (client.public_key, client.secret_key, client.host) == (
        "pk",
        "sk",
        "http://localhost:3000",
    )


def test_v3_spans_updates_events_and_flush() -> None:
    sdk = FakeV3()
    tracer = LangfuseClient(ON, client=sdk)
    with tracer.span("indexing.run", {"docs": 2}) as span:
        with tracer.span("indexing.embed"):
            pass
        span.update(output={"chunks": 5})
    tracer.record("skip", {"document_id": "a"})
    tracer.flush()
    assert sdk.events == [
        ("start", "indexing.run", {"docs": 2}),
        ("start", "indexing.embed", {}),
        ("end", "indexing.embed"),
        ("update", "indexing.run", {"output": {"chunks": 5}}),
        ("end", "indexing.run"),
        ("event", "skip", {"document_id": "a"}),
    ]
    assert sdk.flushed == 1


def test_v2_nested_spans_attach_to_parent() -> None:
    sdk = FakeV2()
    tracer = LangfuseClient(ON, client=sdk)
    with tracer.span("root"):
        with tracer.span("child"):
            pass
    assert ("start", "child", {}, "root") in sdk.events
    assert sdk.events[-1] == ("end", "root")


def test_pipeline_errors_propagate_and_sdk_errors_do_not() -> None:
    sdk = FakeV3()
    tracer = LangfuseClient(ON, client=sdk)
    with pytest.raises(ValueError):
        with tracer.span("boom"):
            raise ValueError("pipeline failure")
    assert sdk.events[-1] == ("end", "boom")

    class Broken:
        def start_as_current_span(self, **_):
            raise RuntimeError("server down")

        def create_event(self, **_):
            raise RuntimeError("server down")

    broken = LangfuseClient(ON, client=Broken())
    with broken.span("x") as span:
        span.update(output=1)
    broken.record("e", {})
