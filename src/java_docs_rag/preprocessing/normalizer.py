"""Developer B: normalize text while preserving Java identifiers and code."""

from java_docs_rag.common.models import Document


class DocumentNormalizer:
    def normalize(self, document: Document) -> Document:
        raise NotImplementedError("Document normalization is not implemented")
