"""Integrated local indexing and question answering; no model/service startup on import."""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from java_docs_rag.common.config import (
    ChunkingConfig,
    EmbeddingConfig,
    ObservabilityConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from java_docs_rag.common.exceptions import JavaDocsRAGError, VectorStoreError
from java_docs_rag.common.interfaces import Embedder
from java_docs_rag.embeddings.factory import create_embedder
from java_docs_rag.generation.ollama_provider import OllamaProvider
from java_docs_rag.observability.langfuse_client import create_tracer
from java_docs_rag.observability.safe_tracing import SafeTelemetry
from java_docs_rag.pipeline.rag_pipeline import RAGPipeline
from java_docs_rag.preprocessing.chunking.base import BaseChunker
from java_docs_rag.preprocessing.chunking.factory import create_chunker
from java_docs_rag.reranking.cross_encoder import CrossEncoderReranker
from java_docs_rag.reranking.sentence_transformer import (
    DEFAULT_RERANKER_MODEL,
    SentenceTransformerScorer,
)
from java_docs_rag.retrieval.retriever import SemanticRetriever
from java_docs_rag.vectorstore.qdrant_store import QdrantVectorStore, collection_name_for


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Java Documentation RAG: index and ask")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("indexing", help="forward arguments to B's grab/prepare/index CLI")
    index = sub.add_parser("index", help="index prepared JSONL or a local JavaDoc HTML sample")
    inputs = index.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="prepared documents JSONL from B's prepare")
    inputs.add_argument("--html", type=Path, help="local JavaDoc HTML; no crawling")
    index.add_argument("--source-url", help="original documentation URL, required with --html")
    index.add_argument(
        "--from-collection", help="seed a NEW alias from a legacy physical collection"
    )
    ask = sub.add_parser("ask", help="answer from the same model/chunking collection")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=_positive, default=10)
    ask.add_argument("--top-n", type=_positive, default=5)
    ask.add_argument("--min-similarity", type=float, default=None)
    ask.add_argument("--max-context-chars", type=int, default=8000)
    ask.add_argument(
        "--reranker-model", default=os.environ.get("RERANKER_MODEL") or DEFAULT_RERANKER_MODEL
    )
    ask.add_argument("--ollama-model", default=os.environ.get("OLLAMA_MODEL"))
    ask.add_argument(
        "--ollama-url", default=os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
    )
    ask.add_argument("--timeout", type=float, default=60.0)
    for command in (index, ask):
        command.add_argument(
            "--trace", action="store_true", help="opt-in count-only Langfuse tracing"
        )
        command.add_argument(
            "--model", default=os.environ.get("EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
        )
        command.add_argument(
            "--dimension",
            type=_positive,
            default=None,
            help="required for embedding models outside B's registry",
        )
        command.add_argument("--device", default=os.environ.get("EMBEDDING_DEVICE") or "cpu")
        command.add_argument("--batch-size", type=_positive, default=64)
        command.add_argument(
            "--strategy", choices=["fixed", "overlap", "paragraph", "section"], default="paragraph"
        )
        command.add_argument("--chunk-size", type=_positive, default=1000)
        command.add_argument("--overlap", type=int, default=150)
        command.add_argument(
            "--qdrant-url", default=os.environ.get("QDRANT_URL") or "http://localhost:6333"
        )
        command.add_argument("--collection", help="exact collection override; otherwise B naming")
    return parser


def _resources(args: argparse.Namespace) -> tuple[BaseChunker, Embedder, QdrantVectorStore]:
    """Resolve the identical B factories and collection naming for index and ask."""
    chunker = create_chunker(ChunkingConfig(args.strategy, args.chunk_size, args.overlap))
    embedder = create_embedder(
        EmbeddingConfig(args.model, args.dimension), batch_size=args.batch_size, device=args.device
    )
    collection = args.collection or collection_name_for(
        os.environ.get("QDRANT_COLLECTION") or "java_docs", embedder.model_id, chunker.config_tag
    )
    store = QdrantVectorStore(
        VectorStoreConfig(args.qdrant_url, collection),
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        extra_payload={"embedding_model": embedder.model_id, "chunking": chunker.config_tag},
    )
    return chunker, embedder, store


def _execute(args: argparse.Namespace, tracer: SafeTelemetry) -> dict:
    store = None
    try:
        if args.command == "index":
            from java_docs_rag.pipeline.collection_rebuild import (
                QdrantCollectionAdmin,
                rebuild_collection,
            )
            from java_docs_rag.pipeline.indexing_pipeline import (
                DocumentPreparer,
                load_documents_jsonl,
            )

            if args.html:
                document = DocumentPreparer().prepare(args.html.read_bytes(), args.source_url)
                if document is None:
                    raise ValueError("HTML sample has too little content")
                documents = [document]
            else:
                documents = load_documents_jsonl(args.input)
            chunker, embedder, store = _resources(args)
            admin = QdrantCollectionAdmin(store.config, api_key=store.api_key)
            try:
                report, build, previous = rebuild_collection(
                    store.collection,
                    documents,
                    chunker,
                    embedder,
                    admin=admin,
                    chunking_tag=chunker.config_tag,
                    batch_size=args.batch_size,
                    from_collection=args.from_collection,
                    tracer=tracer,
                    store_factory=lambda name: QdrantVectorStore(
                        replace(store.config, collection=name),
                        api_key=store.api_key,
                        extra_payload=store.extra_payload,
                    ),
                )
            finally:
                admin.close()
            output = {
                "collection": store.collection,
                "embedding_model": embedder.model_id,
                "dimension": embedder.dimension,
                "build_collection": build,
                "previous_collection": previous,
                **report.as_dict(),
            }
        else:
            provider = OllamaProvider(args.ollama_model, args.ollama_url, args.timeout)
            scorer = SentenceTransformerScorer(
                args.reranker_model, device=args.device, batch_size=args.batch_size
            )
            _, embedder, store = _resources(args)
            size = store.collection_dimension()
            if size != embedder.dimension:
                raise VectorStoreError(
                    f"Collection {store.collection!r} dimension is {size}; expected "
                    f"{embedder.dimension}. Index with the same model/chunking first."
                )
            pipeline = RAGPipeline(
                SemanticRetriever(embedder, store),
                CrossEncoderReranker(scorer.model_id, scorer),
                provider,
                config=RetrievalConfig(args.top_k, args.top_n, args.min_similarity),
                max_context_chars=args.max_context_chars,
                tracer=tracer,
            )
            output = {
                "collection": store.collection,
                "embedding_model": embedder.model_id,
                "dimension": embedder.dimension,
                **asdict(pipeline.run(args.question)),
            }
        return output
    finally:
        if store is not None:
            store.close()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "indexing":
        from java_docs_rag.pipeline.indexing_pipeline import main as indexing_main

        return indexing_main(arguments[1:])
    parser = _parser()
    args = parser.parse_args(arguments)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "index" and args.html and not args.source_url:
        parser.error("--source-url is required with --html")
    if args.command == "ask":
        if not args.ollama_model or not args.ollama_model.strip():
            parser.error("set --ollama-model or OLLAMA_MODEL to an installed Ollama model")
        if args.max_context_chars < 0:
            parser.error("--max-context-chars must be non-negative")
    if args.trace and not all(
        os.environ.get(name)
        for name in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    ):
        parser.error("--trace requires LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    tracer = SafeTelemetry(create_tracer(ObservabilityConfig(enabled=args.trace)))
    try:
        with tracer.span(f"{args.command}.command") as operation:
            output = _execute(args, tracer)
            operation.update(output=output)
            if args.command == "index":
                operation.update(output={"published": True})
            else:
                operation.update(
                    output={
                        "raw_count": len(output["retrieved_chunks"]),
                        "source_count": len(output["sources"]),
                        "outcome": "abstained" if output["insufficient_context"] else "ok",
                    }
                )
    except (JavaDocsRAGError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        tracer.flush()
    if args.trace:
        output["telemetry"] = tracer.summary()
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
