"""Developer B: optional tracing adapter, disabled until explicitly configured."""

from dataclasses import dataclass

from java_docs_rag.common.config import ObservabilityConfig


@dataclass
class LangfuseClient:
    config: ObservabilityConfig

    def record(self, name: str, metadata: dict[str, str]) -> None:
        raise NotImplementedError("Observability is not implemented")
