"""Developer B: future local embedding adapter; no ML libraries imported."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class SentenceTransformerEmbedder:
    model_id: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError("Document embedding is not implemented")

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("Query embedding is not implemented")
