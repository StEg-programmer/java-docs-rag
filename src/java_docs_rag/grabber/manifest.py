"""Developer B: persist source hashes and update metadata for incremental indexing."""

from pathlib import Path

from java_docs_rag.common.models import Document


class DocumentManifest:
    def load(self, path: Path) -> dict[str, str]:
        raise NotImplementedError("Manifest loading is not implemented")

    def save(self, documents: list[Document], path: Path) -> None:
        raise NotImplementedError("Manifest persistence is not implemented")
