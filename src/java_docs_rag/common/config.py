"""Configuration contracts matching configs/default.yaml; no automatic I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: Literal["fixed", "overlap", "paragraph", "section"] = "paragraph"
    chunk_size: int = 1000
    overlap: int = 150


@dataclass(frozen=True)
class EmbeddingConfig:
    model_id: str | None = None
    dimension: int | None = None


@dataclass(frozen=True)
class VectorStoreConfig:
    url: str = "http://localhost:6333"
    collection: str = "java_docs"


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 10
    rerank_top_n: int = 5
    min_similarity: float | None = None


@dataclass(frozen=True)
class GenerationConfig:
    base_url: str = "http://localhost:11434"
    model_id: str | None = None


@dataclass(frozen=True)
class ObservabilityConfig:
    enabled: bool = False


@dataclass(frozen=True)
class AppConfig:
    data_dir: str = "data"
    sources_file: str = "configs/sources.yaml"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)


def load_config(path: str) -> AppConfig:
    """Reserved entry point for future YAML loading and validation."""
    raise NotImplementedError("Configuration loading is not implemented in the scaffold")
