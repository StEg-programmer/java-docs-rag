"""Developer B: remove navigation and other documentation boilerplate."""

from java_docs_rag.common.models import Document


class DocumentCleaner:
    def clean(self, document: Document) -> Document:
        raise NotImplementedError("Document cleaning is not implemented")
