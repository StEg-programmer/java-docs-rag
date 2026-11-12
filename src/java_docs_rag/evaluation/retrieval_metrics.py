"""Developer A: document-level retrieval metrics over unique ranked document IDs."""

from collections.abc import Sequence

from java_docs_rag.common.models import SearchResult


def evaluate_retrieval(
    results: Sequence[SearchResult], relevant_document_ids: Sequence[str], k: int
) -> dict[str, float]:
    """Document-level metrics over the first k unique IDs in first-seen order.

    Precision divides hits by k (missing ranks are misses); recall divides by the
    number of unique relevant IDs. MRR is the reciprocal first relevant rank.
    Empty results or relevance produce zero for all metrics.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be greater than zero")
    ranked = list(dict.fromkeys(result.chunk.document_id for result in results))[:k]
    relevant = set(relevant_document_ids)
    hits = len(relevant.intersection(ranked))
    reciprocal = next((1.0 / rank for rank, doc in enumerate(ranked, 1) if doc in relevant), 0.0)
    return {
        "precision_at_k": hits / k,
        "recall_at_k": hits / len(relevant) if relevant else 0.0,
        "mrr_at_k": reciprocal,
    }
