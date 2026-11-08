"""Developer B: parse downloaded HTML into a common Document."""

from java_docs_rag.common.models import Document


class HTMLDocumentParser:
    def parse(self, html: str, source_url: str) -> Document:
        raise NotImplementedError("HTML parsing is not implemented")
