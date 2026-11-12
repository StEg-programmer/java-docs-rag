"""Developer A: vector retrieval through shared interfaces."""

from dataclasses import dataclass

from java_docs_rag.common.interfaces import Embedder, VectorStore
from java_docs_rag.common.models import SearchResult


@dataclass
class SemanticRetriever:
    """Embed questions and return vector-store results without changing their order or scores."""

    embedder: Embedder
    vectorstore: VectorStore

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        """Retrieve up to top_k results, passing the original query to the embedder.

        Raise TypeError for invalid input types and ValueError for a blank query
        or non-positive top_k. Booleans are not accepted as result counts.
        Validation precedes adapter calls; adapter errors propagate to the caller.
        An empty search returns an empty list.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        vector = self.embedder.embed_query(query)
        return self.vectorstore.search(vector, top_k=top_k)


class VectorRetriever(SemanticRetriever):
    """Backward-compatible name for SemanticRetriever with the same constructor and API."""
