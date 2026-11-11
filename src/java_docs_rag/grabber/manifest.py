"""Developer B: persist source hashes and update metadata for incremental indexing.

Manifest file format (JSON, UTF-8)::

    {"version": 1, "documents": {"<document_id>": {"source_url": ..., "content_hash": ...,
                                                   "updated_at": ..., "extra": {...}}}}
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from java_docs_rag.common.exceptions import DocumentProcessingError
from java_docs_rag.common.models import Document

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ManifestEntry:
    document_id: str
    source_url: str
    content_hash: str
    updated_at: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ManifestDiff:
    added: list[str]
    changed: list[str]
    removed: list[str]
    unchanged: list[str]

    @property
    def to_index(self) -> list[str]:
        return [*self.added, *self.changed]


def diff_hashes(previous: Mapping[str, str], current: Mapping[str, str]) -> ManifestDiff:
    """Compare ``document_id -> content_hash`` mappings (results sorted by ID)."""
    return ManifestDiff(
        added=sorted(k for k in current if k not in previous),
        changed=sorted(k for k in current if k in previous and previous[k] != current[k]),
        removed=sorted(k for k in previous if k not in current),
        unchanged=sorted(k for k in current if previous.get(k) == current[k]),
    )


class DocumentManifest:
    def load_entries(self, path: Path) -> dict[str, ManifestEntry]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DocumentProcessingError(f"Cannot read manifest {path}: {exc}") from exc
        if data.get("version") != MANIFEST_VERSION:
            raise DocumentProcessingError(f"Unsupported manifest version in {path}")
        return {
            doc_id: ManifestEntry(document_id=doc_id, **values)
            for doc_id, values in data.get("documents", {}).items()
        }

    def save_entries(self, entries: Iterable[ManifestEntry], path: Path) -> None:
        documents = {}
        for entry in sorted(entries, key=lambda e: e.document_id):
            values = asdict(entry)
            values.pop("document_id")
            documents[entry.document_id] = values
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {"version": MANIFEST_VERSION, "documents": documents}, indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        tmp.replace(path)

    def load(self, path: Path) -> dict[str, str]:
        """``document_id -> content_hash`` from a manifest (empty if it does not exist)."""
        return {doc_id: e.content_hash for doc_id, e in self.load_entries(path).items()}

    def save(self, documents: list[Document], path: Path) -> None:
        self.save_entries(
            (
                ManifestEntry(
                    document_id=d.document_id,
                    source_url=d.source_url,
                    content_hash=d.content_hash,
                    updated_at=d.updated_at,
                )
                for d in documents
            ),
            path,
        )
