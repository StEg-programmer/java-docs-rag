"""Developer A: document-level retrieval metrics; semantics require an ADR."""

from collections.abc import Sequence

from java_docs_rag.common.models import SearchResult


def evaluate_retrieval(
    results: Sequence[SearchResult], relevant_document_ids: Sequence[str], k: int
) -> dict[str, float]:
    raise NotImplementedError("Retrieval metrics are not implemented")
