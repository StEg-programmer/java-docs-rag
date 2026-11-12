"""Developer B: optional tracing adapter, disabled until explicitly configured.

Pipelines depend on the small ``Tracer`` protocol below, never on the Langfuse SDK:

    tracer = create_tracer(ObservabilityConfig(enabled=True))
    with tracer.span("indexing.chunk", metadata={"strategy": "paragraph"}) as span:
        ...
        span.update(output={"chunks": 42})
    tracer.flush()

* ``NoOpTracer`` — default; records nothing and needs no packages or network.
* ``LangfuseClient`` — imports ``langfuse`` lazily; credentials come from
  ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY`` and ``LANGFUSE_HOST``.
  Works with the v3 SDK (``start_as_current_span``/``start_as_current_observation``)
  and falls back to the v2 ``trace()/span()`` API. Tracing errors are logged and never
  interrupt the pipeline.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from java_docs_rag.common.config import ObservabilityConfig
from java_docs_rag.common.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class SpanHandle(Protocol):
    def update(self, *, output: Any = None, metadata: Mapping[str, Any] | None = None) -> None: ...


class Tracer(Protocol):
    @property
    def enabled(self) -> bool: ...

    def span(
        self, name: str, metadata: Mapping[str, Any] | None = None, input: Any = None
    ) -> Any: ...  # context manager yielding SpanHandle

    def record(self, name: str, metadata: dict[str, str]) -> None: ...

    def flush(self) -> None: ...


class _NoOpSpan:
    def update(self, *, output: Any = None, metadata: Mapping[str, Any] | None = None) -> None:
        return None


class NoOpTracer:
    enabled = False

    @contextmanager
    def span(
        self, name: str, metadata: Mapping[str, Any] | None = None, input: Any = None
    ) -> Iterator[SpanHandle]:
        yield _NoOpSpan()

    def record(self, name: str, metadata: dict[str, str]) -> None:
        return None

    def flush(self) -> None:
        return None


@dataclass
class _SdkSpan:
    native: Any
    updates: dict[str, Any] = field(default_factory=dict)

    def update(self, *, output: Any = None, metadata: Mapping[str, Any] | None = None) -> None:
        fields: dict[str, Any] = {}
        if output is not None:
            fields["output"] = output
        if metadata:
            fields["metadata"] = dict(metadata)
        if not fields:
            return
        self.updates.update(fields)
        try:
            self.native.update(**fields)
        except Exception as exc:  # tracing must not break the pipeline
            logger.warning("Langfuse span update failed (%s)", type(exc).__name__)


@dataclass
class LangfuseClient:
    config: ObservabilityConfig
    public_key: str | None = None
    secret_key: str | None = None
    host: str | None = None
    client: Any = None
    _stack: list[Any] = field(default_factory=list, init=False, repr=False)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @classmethod
    def from_env(cls, config: ObservabilityConfig) -> LangfuseClient:
        return cls(
            config=config,
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY") or None,
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY") or None,
            host=os.environ.get("LANGFUSE_HOST") or None,
        )

    def _sdk(self) -> Any:
        if self.client is None:
            if not (self.public_key and self.secret_key):
                raise ConfigurationError(
                    "Langfuse is enabled but LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set"
                )
            try:
                from langfuse import Langfuse
            except ImportError as exc:
                raise ConfigurationError(
                    'langfuse is not installed; run: pip install -e ".[observability]"'
                ) from exc
            self.client = Langfuse(
                public_key=self.public_key, secret_key=self.secret_key, host=self.host
            )
        return self.client

    @contextmanager
    def _native_span(self, name: str, metadata: dict[str, Any], input: Any) -> Iterator[Any]:
        sdk = self._sdk()
        if hasattr(sdk, "start_as_current_observation"):  # SDK >= 3.x (observation API)
            with sdk.start_as_current_observation(
                as_type="span", name=name, input=input, metadata=metadata
            ) as native:
                yield native
        elif hasattr(sdk, "start_as_current_span"):  # SDK 3.x
            with sdk.start_as_current_span(name=name, input=input, metadata=metadata) as native:
                yield native
        else:  # SDK 2.x: a root span is a trace, nested spans hang off their parent
            parent = self._stack[-1] if self._stack else None
            factory = parent.span if parent is not None else sdk.trace
            native = factory(name=name, input=input, metadata=metadata)
            try:
                yield native
            finally:
                end = getattr(native, "end", None)
                if callable(end):
                    end()

    @contextmanager
    def span(
        self, name: str, metadata: Mapping[str, Any] | None = None, input: Any = None
    ) -> Iterator[SpanHandle]:
        if not self.enabled:
            yield _NoOpSpan()
            return
        try:
            context = self._native_span(name, dict(metadata or {}), input)
            native = context.__enter__()
        except ConfigurationError:
            raise
        except Exception as exc:
            logger.warning("Langfuse span '%s' could not be started (%s)", name, type(exc).__name__)
            yield _NoOpSpan()
            return
        self._stack.append(native)
        error: BaseException | None = None
        try:
            yield _SdkSpan(native)
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._stack.pop()
            try:
                if error is not None:
                    context.__exit__(type(error), error, error.__traceback__)
                else:
                    context.__exit__(None, None, None)
            except Exception as exc:
                logger.warning(
                    "Langfuse span '%s' could not be closed (%s)", name, type(exc).__name__
                )

    def record(self, name: str, metadata: dict[str, str]) -> None:
        """A point-in-time event (e.g. 'document skipped: unchanged')."""
        if not self.enabled:
            return
        try:
            sdk = self._sdk()
            if hasattr(sdk, "create_event"):
                sdk.create_event(name=name, metadata=metadata)
            else:
                sdk.event(name=name, metadata=metadata)
        except ConfigurationError:
            raise
        except Exception as exc:
            logger.warning("Langfuse event '%s' failed (%s)", name, type(exc).__name__)

    def flush(self) -> None:
        if self.client is not None:
            try:
                self.client.flush()
            except Exception as exc:
                logger.warning("Langfuse flush failed (%s)", type(exc).__name__)


def create_tracer(config: ObservabilityConfig) -> NoOpTracer | LangfuseClient:
    """NoOp unless observability is enabled; credentials are read from the environment."""
    if not config.enabled:
        return NoOpTracer()
    return LangfuseClient.from_env(config)
