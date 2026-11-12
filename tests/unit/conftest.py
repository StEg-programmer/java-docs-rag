"""Synthetic evidence for Developer A's filtering, generation and evaluation tests."""

import pytest

from java_docs_rag.common.models import DocumentChunk, SearchResult


@pytest.fixture
def candidates() -> list[SearchResult]:
    return [
        SearchResult(
            DocumentChunk(
                chunk_id=f"chunk-{index}",
                document_id=f"doc-{index}",
                text=text,
                source_url=f"https://example.test/{index}",
                title="Java List",
                section="Description",
                position=index,
                metadata={"version": "21"},
            ),
            similarity_score=score,
        )
        for index, (text, score) in enumerate([("List", 0.9), ("A collection", 0.5), ("Set", 0.1)])
    ]
