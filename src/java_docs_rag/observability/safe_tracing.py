"""Count-only telemetry for the integrated CLI; no business content or exception payloads."""

import logging
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import isfinite
from time import perf_counter
from typing import Any

from java_docs_rag.observability.langfuse_client import NoOpTracer, Tracer

logger = logging.getLogger(__name__)
_COUNTERS = frozenset(
    {
        "documents",
        "documents_total",
        "documents_indexed",
        "documents_skipped",
        "documents_deleted",
        "chunks",
        "batches",
        "embed_seconds",
        "upsert_seconds",
        "duration_ms",
        "top_k",
        "top_n",
        "raw_count",
        "eligible_count",
        "ranked_count",
        "unique_count",
        "source_count",
        "selected_chars",
        "max_context_chars",
        "insufficient_context",
        "provider_called",
        "published",
    }
)
_OUTCOMES = frozenset({"ok", "error", "abstained", "skipped"})


def _safe(fields: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(fields, Mapping):
        return {}
    return {
        key: value
        for key, value in fields.items()
        if (key in _COUNTERS and isinstance(value, (int, float)) and isfinite(value))
        or (key == "outcome" and isinstance(value, str) and value in _OUTCOMES)
    }


@dataclass
class SafeSpan:
    fields: dict[str, Any] = field(default_factory=dict)

    def update(self, *, output: Any = None, metadata: Mapping[str, Any] | None = None) -> None:
        self.fields.update(_safe(output))
        self.fields.update(_safe(metadata))


@dataclass
class SafeTelemetry:
    """Reuse Tracer while isolating its lifecycle failures and sanitizing all fields.

    SDK contexts always close without an exception object: business failures are
    represented only by outcome=error, preventing SDK exception/stack capture.
    Trace IDs are local correlation hints, never proof of server receipt.
    """

    backend: Tracer = field(default_factory=NoOpTracer)
    trace_ids: list[str] = field(default_factory=list, init=False)
    errors: int = field(default=0, init=False)
    _depth: int = field(default=0, init=False)

    @property
    def enabled(self) -> bool:
        return self.backend.enabled

    def _failed(self) -> None:
        self.errors += 1
        logger.warning("Optional telemetry operation failed; pipeline continues")

    def _capture_trace_id(self, handle: Any) -> None:
        try:
            sdk = getattr(self.backend, "client", None)
            getter = getattr(sdk, "get_current_trace_id", None)
            trace_id = (
                getter()
                if callable(getter)
                else getattr(getattr(handle, "native", None), "trace_id", None)
            )
            if isinstance(trace_id, str) and re.fullmatch(r"[a-fA-F0-9-]{16,64}", trace_id):
                if trace_id not in self.trace_ids:
                    self.trace_ids.append(trace_id)
        except Exception:
            self._failed()

    @contextmanager
    def span(
        self, name: str, metadata: Mapping[str, Any] | None = None, input: Any = None
    ) -> Iterator[SafeSpan]:
        started = perf_counter()
        record = SafeSpan(_safe(metadata))
        context = handle = None
        if self.enabled:
            try:
                context = self.backend.span(name, metadata=record.fields.copy(), input=None)
                handle = context.__enter__()
                if self._depth == 0:
                    self._capture_trace_id(handle)
            except Exception:
                context = None
                self._failed()
        self._depth += 1
        try:
            yield record
            record.fields.setdefault("outcome", "ok")
        except BaseException:
            record.fields["outcome"] = "error"
            raise
        finally:
            self._depth -= 1
            record.fields["duration_ms"] = (perf_counter() - started) * 1000
            if handle is not None:
                try:
                    handle.update(output=record.fields.copy())
                except Exception:
                    self._failed()
            if context is not None:
                try:
                    context.__exit__(None, None, None)
                except Exception:
                    self._failed()

    def record(self, name: str, metadata: dict[str, str]) -> None:
        with self.span(name):
            pass

    def flush(self) -> None:
        try:
            self.backend.flush()
        except Exception:
            self._failed()

    def summary(self) -> dict[str, Any]:
        return {"trace_ids": list(self.trace_ids), "errors": self.errors, "receipt_verified": False}
