"""Developer B: explicit embedder selection; never load models on import."""

from __future__ import annotations

from dataclasses import dataclass

from java_docs_rag.common.config import EmbeddingConfig
from java_docs_rag.common.exceptions import ConfigurationError
from java_docs_rag.common.interfaces import Embedder
from java_docs_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder


@dataclass(frozen=True)
class EmbeddingModelSpec:
    model_id: str
    dimension: int
    query_prefix: str = ""
    document_prefix: str = ""
    alias: str = ""


# Mirrors configs/models.yaml. Prefixes follow the model cards on Hugging Face.
KNOWN_MODELS: dict[str, EmbeddingModelSpec] = {
    spec.model_id: spec
    for spec in (
        EmbeddingModelSpec(
            model_id="BAAI/bge-small-en-v1.5",
            dimension=384,
            query_prefix="Represent this sentence for searching relevant passages: ",
            alias="bge-small",
        ),
        EmbeddingModelSpec(
            model_id="intfloat/multilingual-e5-small",
            dimension=384,
            query_prefix="query: ",
            document_prefix="passage: ",
            alias="e5-small",
        ),
    )
}
ALIASES = {spec.alias: model_id for model_id, spec in KNOWN_MODELS.items()}


def resolve_model_spec(model_id: str) -> EmbeddingModelSpec | None:
    return KNOWN_MODELS.get(ALIASES.get(model_id, model_id))


def create_embedder(
    config: EmbeddingConfig,
    *,
    batch_size: int = 32,
    device: str = "cpu",
    cache_folder: str | None = None,
) -> Embedder:
    """Build a lazily-loading embedder; unknown models need an explicit dimension."""
    if not config.model_id:
        raise ConfigurationError("embedding.model_id must be set (e.g. BAAI/bge-small-en-v1.5)")
    spec = resolve_model_spec(config.model_id)
    if spec is None:
        if config.dimension is None:
            raise ConfigurationError(
                f"Unknown embedding model '{config.model_id}': set embedding.dimension"
            )
        spec = EmbeddingModelSpec(model_id=config.model_id, dimension=config.dimension)
    if config.dimension is not None and config.dimension != spec.dimension:
        raise ConfigurationError(
            f"embedding.dimension={config.dimension} does not match {spec.model_id} "
            f"({spec.dimension})"
        )
    return SentenceTransformerEmbedder(
        spec.model_id,
        spec.dimension,
        query_prefix=spec.query_prefix,
        document_prefix=spec.document_prefix,
        batch_size=batch_size,
        device=device,
        cache_folder=cache_folder,
    )
