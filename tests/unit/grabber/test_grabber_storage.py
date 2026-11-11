"""Developer B: raw repository paths and manifest persistence/diffing."""

from pathlib import Path

import pytest

from java_docs_rag.common.exceptions import DocumentProcessingError
from java_docs_rag.common.models import Document
from java_docs_rag.grabber.manifest import DocumentManifest, ManifestEntry, diff_hashes
from java_docs_rag.grabber.repository import DocumentRepository


def test_repository_round_trip_and_layout(tmp_path: Path) -> None:
    repo = DocumentRepository(tmp_path)
    path = repo.save("javase21:java.base/java/util/List", b"<html/>")
    assert path == tmp_path / "javase21" / "java.base" / "java" / "util" / "List.html"
    assert repo.load("javase21:java.base/java/util/List") == b"<html/>"
    assert repo.exists("javase21:java.base/java/util/List")
    assert not repo.exists("javase21:java.base/java/util/Map")


@pytest.mark.parametrize("bad", ["no-prefix", "p:../../etc/passwd", ":x", "p:"])
def test_repository_rejects_unsafe_ids(tmp_path: Path, bad: str) -> None:
    with pytest.raises(DocumentProcessingError):
        DocumentRepository(tmp_path).save(bad, b"")


def test_repository_missing_document(tmp_path: Path) -> None:
    with pytest.raises(DocumentProcessingError):
        DocumentRepository(tmp_path).load("p:missing")


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = DocumentManifest()
    path = tmp_path / "m" / "index.json"
    assert manifest.load(path) == {}
    docs = [
        Document("b", "https://x/b", "B", "text", "hb"),
        Document("a", "https://x/a", "A", "text", "ha", "2023-09-11"),
    ]
    manifest.save(docs, path)
    assert manifest.load(path) == {"a": "ha", "b": "hb"}
    entries = manifest.load_entries(path)
    assert entries["a"] == ManifestEntry("a", "https://x/a", "ha", "2023-09-11")


def test_manifest_rejects_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DocumentProcessingError):
        DocumentManifest().load(path)


def test_diff_hashes() -> None:
    diff = diff_hashes({"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "X", "d": "4"})
    assert diff.added == ["d"] and diff.changed == ["b"]
    assert diff.removed == ["c"] and diff.unchanged == ["a"]
    assert diff.to_index == ["d", "b"]
