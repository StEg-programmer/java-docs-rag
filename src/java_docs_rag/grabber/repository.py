"""Developer B: local raw-document repository (files under data/raw)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from java_docs_rag.common.exceptions import DocumentProcessingError

_UNSAFE = re.compile(r'[<>:"\\|?*\x00-\x1f]')


@dataclass
class DocumentRepository:
    """Maps ``prefix:relative/path`` document IDs to ``root/prefix/relative/path.html``."""

    root: Path
    suffix: str = ".html"

    def path_for(self, document_id: str) -> Path:
        prefix, sep, rest = document_id.partition(":")
        if not sep or not prefix or not rest:
            raise DocumentProcessingError(f"Invalid document_id '{document_id}'")
        parts = [prefix, *PurePosixPath(rest).parts]
        if any(part in {"..", "."} or part.startswith("/") for part in parts):
            raise DocumentProcessingError(f"Unsafe document_id '{document_id}'")
        safe = [_UNSAFE.sub("_", part) for part in parts]
        return self.root.joinpath(*safe[:-1], safe[-1] + self.suffix)

    def save(self, document_id: str, content: bytes) -> Path:
        path = self.path_for(document_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def load(self, document_id: str) -> bytes:
        path = self.path_for(document_id)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise DocumentProcessingError(f"Raw document '{document_id}' is not stored") from exc

    def exists(self, document_id: str) -> bool:
        return self.path_for(document_id).is_file()
