"""Developer B: indexer, in-memory store and end-to-end indexing pipeline (all offline)."""

import json
from pathlib import Path

import pytest
from conftest import FakeEmbedder, make_document

from java_docs_rag.common.config import ChunkingConfig, ObservabilityConfig
from java_docs_rag.common.exceptions import (
    DocumentProcessingError,
    EmbeddingError,
    VectorStoreError,
)
from java_docs_rag.grabber.crawler import DocumentationCrawler
from java_docs_rag.grabber.downloader import FetchResult
from java_docs_rag.grabber.manifest import DocumentManifest
from java_docs_rag.grabber.repository import DocumentRepository
from java_docs_rag.grabber.sources import SourceConfig
from java_docs_rag.observability.langfuse_client import LangfuseClient
from java_docs_rag.pipeline.indexing_pipeline import (
    IndexingPipeline,
    corpus_version,
    grab_source,
    load_documents_jsonl,
    main,
    prepare_documents,
    save_documents_jsonl,
)
from java_docs_rag.preprocessing.chunking.factory import create_chunker
from java_docs_rag.vectorstore.indexer import DocumentIndexer, embedding_text
from java_docs_rag.vectorstore.memory import InMemoryVectorStore

FIXTURE = Path("tests/fixtures/javadoc/java.base/java/util/List.html")
API = "https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/"


def pipeline(embedder, store=None, **kwargs) -> IndexingPipeline:
    chunker = create_chunker(ChunkingConfig("paragraph", 200, 0))
    return IndexingPipeline(
        chunker, embedder, InMemoryVectorStore() if store is None else store, batch_size=2, **kwargs
    )


def test_memory_store_search_filters_and_delete() -> None:
    store = InMemoryVectorStore()
    docs = [make_document("aaa", "aaa"), make_document("bbb", "bbb")]
    chunks = [create_chunker(ChunkingConfig("fixed", 50, 0)).split(d)[0] for d in docs]
    store.upsert(chunks, [[1.0, 0.0], [0.0, 2.0]])
    assert [r.chunk.text for r in store.search([0.0, 1.0], 1)] == ["bbb"]
    assert store.search([0.0, 1.0], 2)[0].similarity_score == pytest.approx(1.0)
    assert len(store.search([1.0, 1.0], 5, {"document_id": chunks[0].document_id})) == 1
    assert store.search([1.0, 1.0], 5, {"java_version": "17"}) == []
    store.delete_document(chunks[0].document_id)
    assert len(store) == 1
    with pytest.raises(VectorStoreError):
        store.search([1.0], 1)


def test_indexer_batches_and_adds_context(embedder: FakeEmbedder) -> None:
    doc = make_document("list", "\n\n".join(f"Paragraph number {i} text." for i in range(5)))
    chunks = create_chunker(ChunkingConfig("paragraph", 30, 0)).split(doc)
    store = InMemoryVectorStore()
    stats = DocumentIndexer(embedder, store, batch_size=2).index(chunks)
    assert stats.chunks == 5 and stats.batches == 3 and len(store) == 5
    assert [len(call) for call in embedder.calls] == [2, 2, 1]
    assert embedder.calls[0][0] == "List\nParagraph number 0 text."
    assert embedding_text(chunks[0], include_context=False) == chunks[0].text


def test_indexer_rejects_wrong_vector_count() -> None:
    class Broken(FakeEmbedder):
        def embed_documents(self, texts):
            return []

    chunks = create_chunker(ChunkingConfig("fixed", 50, 0)).split(make_document("a", "abc"))
    with pytest.raises(EmbeddingError):
        DocumentIndexer(Broken(), InMemoryVectorStore()).index(chunks)


def test_pipeline_indexes_and_is_incremental(embedder: FakeEmbedder) -> None:
    store = InMemoryVectorStore()
    docs = [make_document("list", "Ordered collection. " * 20), make_document("map", "Keys.")]
    first = pipeline(embedder, store).run(docs)
    assert first.documents_indexed == 2 and first.chunks == len(store) > 2

    changed = [make_document("list", "Short now."), docs[1], make_document("set", "Unique.")]
    second = pipeline(embedder, store).run(
        changed, previous_hashes=first.hashes, delete_missing=True
    )
    assert (second.documents_indexed, second.documents_skipped) == (2, 1)
    texts = sorted(r.chunk.text for r in store.search([1.0] * 26, 50))
    assert texts == ["Keys.", "Short now.", "Unique."]  # no stale "list" chunks

    third = pipeline(embedder, store).run(
        [changed[0]], previous_hashes=second.hashes, delete_missing=True
    )
    assert third.documents_deleted == 2 and len(store) == 1


def test_pipeline_traces_when_enabled(embedder: FakeEmbedder) -> None:
    from test_tracing_support import RecordingSdk

    sdk = RecordingSdk()
    tracer = LangfuseClient(ObservabilityConfig(enabled=True), client=sdk)
    pipeline(embedder, tracer=tracer).run([make_document("a", "alpha beta")])
    assert [name for kind, name in sdk.events if kind == "start"] == [
        "indexing.run",
        "indexing.embed_and_upsert",
    ]
    assert sdk.flushed


class Site:
    def __init__(self, pages: dict[str, bytes]) -> None:
        self.pages = pages

    def fetch(self, url: str) -> FetchResult:
        if url not in self.pages:
            raise DocumentProcessingError(f"HTTP 404 {url}")
        return FetchResult(url, url, 200, self.pages[url], "text/html")


def test_grab_prepare_index_end_to_end(tmp_path: Path, embedder: FakeEmbedder) -> None:
    html = FIXTURE.read_bytes()
    stub = b"<html><body><main><p>tiny</p></main></body></html>"
    site = Site(
        {
            API + "package-summary.html": b'<a href="List.html">L</a><a href="X.html">X</a>'
            + b"<main><p>"
            + b"The java.util package. " * 5
            + b"</p></main>",
            API + "List.html": html,
            API + "X.html": stub,
        }
    )
    source = SourceConfig(
        "test",
        "Test",
        API,
        ["docs.oracle.com"],
        allowed_path_prefixes=["/en/java/javase/21/docs/api/java.base/java/util/"],
        start_urls=[API + "package-summary.html"],
    )
    repo = DocumentRepository(tmp_path / "raw")
    raw_manifest = tmp_path / "manifests" / "test.raw.json"
    report = grab_source(
        source, repo, raw_manifest, DocumentationCrawler(site, respect_robots=False)
    )
    assert (report.fetched, report.changed) == (3, 3)
    assert API + "Collection.html" in report.failures  # linked from List.html, not served
    again = grab_source(
        source, repo, raw_manifest, DocumentationCrawler(site, respect_robots=False)
    )
    assert again.changed == 0

    documents, skipped = prepare_documents(repo, raw_manifest)
    assert [d.document_id for d in documents] == [
        "javase21:java.base/java/util/List",
        "javase21:java.base/java/util/package-summary",
    ]
    assert skipped == {"javase21:java.base/java/util/X": "too little content"}

    processed = tmp_path / "processed" / "test.jsonl"
    save_documents_jsonl(documents, processed)
    assert load_documents_jsonl(processed) == documents
    assert corpus_version(documents) == corpus_version(reversed(documents))

    store = InMemoryVectorStore()
    result = pipeline(embedder, store).run(documents)
    assert result.chunks == len(store)
    hit = store.search(embedder.embed_query("number of elements in this list size"), 1)[0]
    assert hit.chunk.document_id == "javase21:java.base/java/util/List"
    assert hit.chunk.source_url == API + "List.html"


def test_cli_prepare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = DocumentRepository(tmp_path / "raw")
    repo.save("javase21:java.base/java/util/List", FIXTURE.read_bytes())
    from java_docs_rag.grabber.manifest import ManifestEntry

    DocumentManifest().save_entries(
        [ManifestEntry("javase21:java.base/java/util/List", API + "List.html", "h")],
        tmp_path / "manifests" / "s.raw.json",
    )
    assert main(["--data-dir", str(tmp_path), "--source", "s", "prepare"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["documents"] == 1 and len(output["corpus_version"]) == 16
    assert main(["--data-dir", str(tmp_path / "empty"), "--source", "s", "prepare"]) == 1
