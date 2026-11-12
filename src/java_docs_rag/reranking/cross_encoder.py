"""Developer A: cross-encoder reranking through an injected batch scorer."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from java_docs_rag.common.models import SearchResult


@dataclass
class CrossEncoderReranker:
    """Use a caller-provided scorer without loading models or performing I/O."""

    model_id: str
    scorer: Callable[[Sequence[tuple[str, str]]], Sequence[float]]

    def rerank(self, query: str, results: Sequence[SearchResult], top_n: int) -> list[SearchResult]:
        """Score all candidates and return new results in stable descending score order.

        Raise TypeError for non-integer or boolean top_n, and ValueError for
        non-positive top_n or a scorer output count that differs from the input.
        Empty input returns an empty list without calling the scorer.
        """
        if isinstance(top_n, bool) or not isinstance(top_n, int):
            raise TypeError("top_n must be an integer")
        if top_n <= 0:
            raise ValueError("top_n must be greater than zero")
        if not results:
            return []

        scores = self.scorer([(query, result.chunk.text) for result in results])
        if len(scores) != len(results):
            raise ValueError(
                f"scorer returned {len(scores)} scores for {len(results)} candidates; "
                "expected one score per candidate"
            )

        ranked = sorted(zip(results, scores, strict=True), key=lambda item: item[1], reverse=True)
        return [replace(result, rerank_score=score) for result, score in ranked[:top_n]]
