"""Developer B: explicit embedder selection; never load models on import."""

from java_docs_rag.common.config import EmbeddingConfig
from java_docs_rag.common.interfaces import Embedder


def create_embedder(config: EmbeddingConfig) -> Embedder:
    raise NotImplementedError("Embedder selection is not implemented")
