"""Developer A: future local cross-encoder adapter; no model loading."""

from collections.abc import Sequence
from dataclasses import dataclass

from java_docs_rag.common.models import SearchResult


@dataclass
class CrossEncoderReranker:
    model_id: str

    def rerank(self, query: str, results: Sequence[SearchResult], top_n: int) -> list[SearchResult]:
        raise NotImplementedError("Reranking is not implemented")
