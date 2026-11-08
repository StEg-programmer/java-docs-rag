"""Developer A: format selected chunks with traceable source identifiers."""

from collections.abc import Sequence

from java_docs_rag.common.models import DocumentChunk


class ContextBuilder:
    def build(self, chunks: Sequence[DocumentChunk]) -> str:
        raise NotImplementedError("Context building is not implemented")
