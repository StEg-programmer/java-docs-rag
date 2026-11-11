"""Developer B: indexing orchestration.

Three stages, each usable from Python and from the command line::

    python -m java_docs_rag.pipeline.indexing_pipeline grab    --source java-se-21-api
    python -m java_docs_rag.pipeline.indexing_pipeline prepare --source java-se-21-api
    python -m java_docs_rag.pipeline.indexing_pipeline index   --source java-se-21-api \
        --model BAAI/bge-small-en-v1.5 --strategy paragraph --chunk-size 1000

* grab    — crawl the source, store raw HTML in data/raw, write data/manifests/<id>.raw.json
* prepare — parse/clean/normalize raw pages into data/processed/<id>.jsonl
* index   — chunk, embed and upsert into Qdrant; unchanged documents are skipped
            using data/manifests/index__<collection>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from java_docs_rag.common.config import ChunkingConfig, EmbeddingConfig, VectorStoreConfig
from java_docs_rag.common.exceptions import DocumentProcessingError, JavaDocsRAGError
from java_docs_rag.common.interfaces import Chunker, Embedder, VectorStore
from java_docs_rag.common.models import Document
from java_docs_rag.grabber.crawler import DocumentationCrawler
from java_docs_rag.grabber.identity import content_hash, document_id_from_url
from java_docs_rag.grabber.manifest import DocumentManifest, ManifestEntry, diff_hashes
from java_docs_rag.grabber.repository import DocumentRepository
from java_docs_rag.grabber.sources import SourceConfig
from java_docs_rag.observability.langfuse_client import NoOpTracer, Tracer
from java_docs_rag.preprocessing.cleaner import DocumentCleaner
from java_docs_rag.preprocessing.normalizer import DocumentNormalizer
from java_docs_rag.preprocessing.parser import HTMLDocumentParser
from java_docs_rag.vectorstore.indexer import DocumentIndexer, IndexingStats

logger = logging.getLogger(__name__)


# -- reports ----------------------------------------------------------------------------


@dataclass
class GrabReport:
    source_id: str
    fetched: int = 0
    changed: int = 0
    failures: dict[str, str] = field(default_factory=dict)


@dataclass
class IndexingReport:
    documents_total: int = 0
    documents_indexed: int = 0
    documents_skipped: int = 0
    documents_deleted: int = 0
    chunks: int = 0
    stats: IndexingStats = field(default_factory=IndexingStats)
    hashes: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("hashes")
        data["stats"]["total_seconds"] = self.stats.total_seconds
        return data


def corpus_version(documents: Iterable[Document]) -> str:
    """Short fingerprint of a prepared corpus (IDs + content hashes)."""
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda d: d.document_id):
        digest.update(f"{document.document_id}\t{document.content_hash}\n".encode())
    return digest.hexdigest()[:16]


# -- stage 1: grab --------------------------------------------------------------------


def grab_source(
    source: SourceConfig,
    repository: DocumentRepository,
    manifest_path: Path,
    crawler: DocumentationCrawler,
    tracer: Tracer | None = None,
) -> GrabReport:
    tracer = tracer or NoOpTracer()
    manifest = DocumentManifest()
    previous = manifest.load_entries(manifest_path)
    entries: dict[str, ManifestEntry] = {}
    report = GrabReport(source.source_id)
    with tracer.span("indexing.grab", metadata={"source_id": source.source_id}) as span:
        for page in crawler.iter_pages(
            source.base_url, source.allowed_domains, source.allowed_path_prefixes, source.seeds
        ):
            document_id = document_id_from_url(page.final_url)
            raw_hash = content_hash(page.content)
            repository.save(document_id, page.content)
            old = previous.get(document_id)
            entries[document_id] = ManifestEntry(
                document_id=document_id,
                source_url=page.final_url,
                content_hash=raw_hash,
                updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
                extra={"source_id": source.source_id},
            )
            report.fetched += 1
            if old is None or old.content_hash != raw_hash:
                report.changed += 1
            if report.fetched % 50 == 0:
                logger.info("fetched %d pages", report.fetched)
        report.failures = dict(crawler.failures)
        span.update(
            output={
                "fetched": report.fetched,
                "changed": report.changed,
                "failures": len(report.failures),
            }
        )
    manifest.save_entries(entries.values(), manifest_path)
    return report


# -- stage 2: prepare -----------------------------------------------------------------


@dataclass
class DocumentPreparer:
    parser: HTMLDocumentParser = field(default_factory=HTMLDocumentParser)
    cleaner: DocumentCleaner = field(default_factory=DocumentCleaner)
    normalizer: DocumentNormalizer = field(default_factory=DocumentNormalizer)
    min_content_chars: int = 50

    def prepare(self, html: bytes | str, source_url: str) -> Document | None:
        document = self.parser.parse(html, source_url)
        document = self.normalizer.normalize(self.cleaner.clean(document))
        if len(document.content) < self.min_content_chars:
            return None
        return document


def prepare_documents(
    repository: DocumentRepository,
    raw_manifest_path: Path,
    preparer: DocumentPreparer | None = None,
    tracer: Tracer | None = None,
) -> tuple[list[Document], dict[str, str]]:
    """Returns prepared documents and ``{document_id: reason}`` for skipped pages."""
    preparer = preparer or DocumentPreparer()
    tracer = tracer or NoOpTracer()
    entries = DocumentManifest().load_entries(raw_manifest_path)
    if not entries:
        raise DocumentProcessingError(f"No raw documents listed in {raw_manifest_path}; run grab")
    documents: list[Document] = []
    skipped: dict[str, str] = {}
    with tracer.span("indexing.prepare", metadata={"raw_documents": len(entries)}) as span:
        for document_id, entry in sorted(entries.items()):
            try:
                document = preparer.prepare(repository.load(document_id), entry.source_url)
            except DocumentProcessingError as exc:
                skipped[document_id] = str(exc)
                continue
            if document is None:
                skipped[document_id] = "too little content"
                continue
            documents.append(document)
        span.update(output={"documents": len(documents), "skipped": len(skipped)})
    return documents, skipped


def save_documents_jsonl(documents: Sequence[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")


def load_documents_jsonl(path: Path) -> list[Document]:
    if not path.exists():
        raise DocumentProcessingError(f"{path} does not exist; run prepare")
    with path.open(encoding="utf-8") as handle:
        return [Document(**json.loads(line)) for line in handle if line.strip()]


# -- stage 3: index -------------------------------------------------------------------


@dataclass
class IndexingPipeline:
    chunker: Chunker
    embedder: Embedder
    vectorstore: VectorStore
    batch_size: int = 64
    include_context: bool = True
    tracer: Tracer = field(default_factory=NoOpTracer)

    def run(
        self,
        documents: Sequence[Document],
        previous_hashes: Mapping[str, str] | None = None,
        delete_missing: bool = False,
    ) -> IndexingReport:
        """Index ``documents``. With ``previous_hashes`` (from the index manifest) only
        new/changed documents are re-embedded; stale chunks of a changed document are
        deleted first so a shorter new version leaves no orphans."""
        previous = dict(previous_hashes or {})
        current = {d.document_id: d.content_hash for d in documents}
        diff = diff_hashes(previous, current)
        todo = set(diff.to_index) if previous_hashes is not None else set(current)
        report = IndexingReport(documents_total=len(documents), hashes=current)
        indexer = DocumentIndexer(
            self.embedder, self.vectorstore, self.batch_size, self.include_context
        )
        metadata = {
            "embedding_model": self.embedder.model_id,
            "documents": len(documents),
            "to_index": len(todo),
            "chunker": getattr(self.chunker, "config_tag", type(self.chunker).__name__),
        }
        with self.tracer.span("indexing.run", metadata=metadata) as root:
            if delete_missing:
                for document_id in diff.removed:
                    self.vectorstore.delete_document(document_id)
                    report.documents_deleted += 1
            pending = []
            for document in documents:
                if document.document_id not in todo:
                    report.documents_skipped += 1
                    continue
                if document.document_id in previous:
                    self.vectorstore.delete_document(document.document_id)
                pending.extend(self.chunker.split(document))
                report.documents_indexed += 1
            with self.tracer.span(
                "indexing.embed_and_upsert", metadata={"chunks": len(pending)}
            ) as span:
                stats = indexer.index(pending)
                span.update(
                    output={
                        "batches": stats.batches,
                        "embed_seconds": stats.embed_seconds,
                        "upsert_seconds": stats.upsert_seconds,
                    }
                )
            report.chunks = stats.chunks
            report.stats = stats
            root.update(output=report.as_dict())
        self.tracer.flush()
        return report


# -- command line ---------------------------------------------------------------------


def _paths(data_dir: Path, source_id: str) -> dict[str, Path]:
    return {
        "raw": data_dir / "raw",
        "raw_manifest": data_dir / "manifests" / f"{source_id}.raw.json",
        "processed": data_dir / "processed" / f"{source_id}.jsonl",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Java docs indexing pipeline (Developer B)")
    parser.add_argument("--sources", default="configs/sources.yaml", type=Path)
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--source", default="java-se-21-api")
    parser.add_argument("--trace", action="store_true", help="send traces to Langfuse")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    grab = sub.add_parser("grab", help="crawl and store raw HTML")
    grab.add_argument("--max-pages", type=int, default=None)
    grab.add_argument("--delay", type=float, default=0.5, help="seconds between requests")

    sub.add_parser("prepare", help="parse, clean and normalize raw HTML")

    index = sub.add_parser("index", help="chunk, embed and store in Qdrant")
    index.add_argument(
        "--model", default=None, help="default: $EMBEDDING_MODEL or BAAI/bge-small-en-v1.5"
    )
    index.add_argument(
        "--strategy", default="paragraph", choices=["fixed", "overlap", "paragraph", "section"]
    )
    index.add_argument("--chunk-size", type=int, default=1000)
    index.add_argument("--overlap", type=int, default=150)
    index.add_argument("--qdrant-url", default=None)
    index.add_argument(
        "--collection", default=None, help="defaults to <QDRANT_COLLECTION>__<model>__<chunking>"
    )
    index.add_argument("--batch-size", type=int, default=64)
    index.add_argument("--device", default=None, help="default: $EMBEDDING_DEVICE or cpu")
    index.add_argument(
        "--no-context", action="store_true", help="embed chunk text without title/section header"
    )
    index.add_argument("--full", action="store_true", help="re-index unchanged documents too")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import os

    from java_docs_rag.common.config import ObservabilityConfig
    from java_docs_rag.embeddings.factory import create_embedder
    from java_docs_rag.grabber.downloader import DocumentDownloader
    from java_docs_rag.grabber.sources import get_source, load_sources
    from java_docs_rag.observability.langfuse_client import create_tracer
    from java_docs_rag.preprocessing.chunking.factory import create_chunker
    from java_docs_rag.vectorstore.qdrant_store import QdrantVectorStore, collection_name_for

    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    tracer = create_tracer(ObservabilityConfig(enabled=args.trace))
    paths = _paths(args.data_dir, args.source)
    try:
        if args.command == "grab":
            source = get_source(load_sources(args.sources), args.source)
            downloader = DocumentDownloader(request_delay=args.delay)
            crawler = DocumentationCrawler(
                downloader,
                max_pages=args.max_pages or source.max_pages,
                exclude_patterns=source.exclude_patterns,
            )
            report = grab_source(
                source, DocumentRepository(paths["raw"]), paths["raw_manifest"], crawler, tracer
            )
            downloader.close()
            print(
                json.dumps(
                    {
                        "fetched": report.fetched,
                        "changed": report.changed,
                        "failures": len(report.failures),
                    },
                    indent=2,
                )
            )
            for url, reason in list(report.failures.items())[:20]:
                print(f"  failed: {url}: {reason}", file=sys.stderr)
        elif args.command == "prepare":
            documents, skipped = prepare_documents(
                DocumentRepository(paths["raw"]), paths["raw_manifest"], tracer=tracer
            )
            save_documents_jsonl(documents, paths["processed"])
            print(
                json.dumps(
                    {
                        "documents": len(documents),
                        "skipped": len(skipped),
                        "corpus_version": corpus_version(documents),
                        "output": str(paths["processed"]),
                    },
                    indent=2,
                )
            )
        else:
            documents = load_documents_jsonl(paths["processed"])
            chunking = ChunkingConfig(args.strategy, args.chunk_size, args.overlap)
            chunker = create_chunker(chunking)
            model = args.model or os.environ.get("EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
            device = args.device or os.environ.get("EMBEDDING_DEVICE") or "cpu"
            embedder = create_embedder(
                EmbeddingConfig(model), batch_size=args.batch_size, device=device
            )
            base = os.environ.get("QDRANT_COLLECTION", "java_docs")
            collection = args.collection or collection_name_for(
                base, embedder.model_id, chunker.config_tag
            )
            url = args.qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
            store = QdrantVectorStore(
                VectorStoreConfig(url=url, collection=collection),
                api_key=os.environ.get("QDRANT_API_KEY") or None,
                extra_payload={
                    "embedding_model": embedder.model_id,
                    "chunking": chunker.config_tag,
                },
            )
            index_manifest = args.data_dir / "manifests" / f"index__{collection}.json"
            manifest = DocumentManifest()
            previous = None if args.full else manifest.load(index_manifest)
            pipeline = IndexingPipeline(
                chunker, embedder, store, args.batch_size, not args.no_context, tracer
            )
            report = pipeline.run(documents, previous_hashes=previous, delete_missing=True)
            manifest.save(documents, index_manifest)
            print(
                json.dumps(
                    {
                        "collection": collection,
                        "corpus_version": corpus_version(documents),
                        **report.as_dict(),
                        "points_in_collection": store.count(),
                    },
                    indent=2,
                    default=str,
                )
            )
    except JavaDocsRAGError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        tracer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
