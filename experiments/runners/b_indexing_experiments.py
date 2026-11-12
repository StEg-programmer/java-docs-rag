"""Developer B experiments: chunking strategy / chunk size / overlap / embedding model.

Usage (from the repository root, after `indexing_pipeline grab` and `prepare`)::

    python experiments/runners/b_indexing_experiments.py --dry-run
    python experiments/runners/b_indexing_experiments.py --group chunk_size
    python experiments/runners/b_indexing_experiments.py --store qdrant --limit-docs 200

Every run writes ``experiments/results/<experiment_id>.json`` and appends a row to
``experiments/results/b_indexing_summary.csv``. Indexing metrics (chunk count, sizes,
timings) are always measured. Retrieval metrics come from Developer A's
``java_docs_rag.evaluation`` package; while it is not implemented the result contains
``"retrieval": null`` with the reason — values are never estimated or invented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from java_docs_rag.common.config import ChunkingConfig, EmbeddingConfig, VectorStoreConfig
from java_docs_rag.common.interfaces import Embedder, VectorStore
from java_docs_rag.common.models import Document, SearchResult
from java_docs_rag.pipeline.indexing_pipeline import corpus_version, load_documents_jsonl
from java_docs_rag.preprocessing.chunking.factory import create_chunker
from java_docs_rag.vectorstore.indexer import DocumentIndexer


@dataclass(frozen=True)
class RunConfig:
    group: str
    model: str
    strategy: str
    chunk_size: int
    overlap: int

    @property
    def chunking(self) -> ChunkingConfig:
        return ChunkingConfig(self.strategy, self.chunk_size, self.overlap)


def expand_plan(plan: dict[str, Any], only_group: str | None = None) -> list[RunConfig]:
    """One RunConfig per value of each group; unrelated parameters stay at baseline."""
    base = plan["baseline"]
    runs: list[RunConfig] = []
    for group, spec in plan["groups"].items():
        if only_group and group != only_group:
            continue
        for value in spec["values"]:
            params = {**base, **spec.get("overrides", {})}
            vary = spec["vary"]
            if vary == "overlap_percent":
                params["overlap"] = round(params["chunk_size"] * int(value) / 100)
            else:
                params[vary] = value
            if params["strategy"] != "overlap":
                params["overlap"] = 0
            runs.append(
                RunConfig(
                    group,
                    params["model"],
                    params["strategy"],
                    int(params["chunk_size"]),
                    int(params["overlap"]),
                )
            )
    return runs


def _file_sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None


def _alias(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace(".", "-")


def evaluate_retrieval_quality(
    embedder: Embedder, store: VectorStore, dataset_path: Path, top_ks: Sequence[int]
) -> tuple[dict[str, Any] | None, str | None]:
    """Uses Developer A's dataset loader and metrics. Returns (metrics, reason_if_none)."""
    if not dataset_path.exists():
        return None, f"evaluation dataset {dataset_path} not found"
    try:
        from java_docs_rag.evaluation.dataset import load_dataset
        from java_docs_rag.evaluation.retrieval_metrics import evaluate_retrieval

        questions = [
            q for q in load_dataset(dataset_path) if q.answerable and q.relevant_document_ids
        ]
        if not questions:
            return None, "evaluation dataset has no answerable questions with relevant documents"
        per_k: dict[str, Any] = {}
        searches: dict[str, list[SearchResult]] = {}
        latency: list[float] = []
        for question in questions:
            started = time.perf_counter()
            searches[question.question_id] = store.search(
                embedder.embed_query(question.question), max(top_ks)
            )
            latency.append(time.perf_counter() - started)
        for k in top_ks:
            rows = [
                # A's document-level metric applies k after deduplicating document IDs.
                evaluate_retrieval(searches[q.question_id], q.relevant_document_ids, k)
                for q in questions
            ]
            per_k[str(k)] = {name: statistics.fmean(row[name] for row in rows) for name in rows[0]}
        return {
            "questions": len(questions),
            "metrics_at_k": per_k,
            "query_latency_ms_mean": 1000 * statistics.fmean(latency),
        }, None
    except NotImplementedError as exc:
        return None, f"pending Developer A evaluation framework: {exc}"


def run_one(
    run: RunConfig,
    documents: Sequence[Document],
    make_embedder: Callable[[str], Embedder],
    make_store: Callable[[str, str], VectorStore],
    dataset_path: Path,
    top_ks: Sequence[int],
    batch_size: int = 64,
) -> dict[str, Any]:
    chunker = create_chunker(run.chunking)
    started = time.perf_counter()
    chunks = [chunk for document in documents for chunk in chunker.split(document)]
    chunk_seconds = time.perf_counter() - started
    lengths = [len(c.text) for c in chunks] or [0]

    embedder = make_embedder(run.model)
    store = make_store(run.model, chunker.config_tag)
    stats = DocumentIndexer(embedder, store, batch_size=batch_size).index(chunks)
    retrieval, reason = evaluate_retrieval_quality(embedder, store, dataset_path, top_ks)

    version = corpus_version(documents)
    experiment_id = f"b-{run.group}-{chunker.config_tag}-{_alias(run.model)}-{version[:8]}"
    return {
        "experiment_id": experiment_id,
        "owner": "developer-b",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "group": run.group,
        "conditions": {
            "corpus_version": version,
            "documents": len(documents),
            "embedding_model": run.model,
            "embedding_dimension": embedder.dimension,
            "chunking": {
                "strategy": run.strategy,
                "chunk_size": run.chunk_size,
                "overlap": run.overlap,
                "unit": "characters",
            },
            "vector_collection": getattr(store, "collection", type(store).__name__),
            "retrieval": {"top_k": list(top_ks), "distance": "cosine"},
            "reranker": None,
            "llm": None,
            "prompt_version": None,
            "evaluation_dataset": str(dataset_path),
            "evaluation_dataset_sha": _file_sha(dataset_path),
        },
        "indexing": {
            "chunks": len(chunks),
            "chunk_chars_mean": statistics.fmean(lengths),
            "chunk_chars_median": statistics.median(lengths),
            "chunk_chars_max": max(lengths),
            "chunking_seconds": chunk_seconds,
            "embed_seconds": stats.embed_seconds,
            "upsert_seconds": stats.upsert_seconds,
            "chunks_per_second": stats.chunks / stats.embed_seconds
            if stats.embed_seconds
            else None,
        },
        "retrieval": retrieval,
        "retrieval_unavailable_reason": reason,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }


SUMMARY_FIELDS = [
    "experiment_id",
    "group",
    "embedding_model",
    "strategy",
    "chunk_size",
    "overlap",
    "documents",
    "chunks",
    "embed_seconds",
    "retrieval_available",
]


def save_result(result: dict[str, Any], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{result['experiment_id']}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = results_dir / "b_indexing_summary.csv"
    new = not summary.exists()
    with summary.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if new:
            writer.writeheader()
        c = result["conditions"]
        writer.writerow(
            {
                "experiment_id": result["experiment_id"],
                "group": result["group"],
                "embedding_model": c["embedding_model"],
                "strategy": c["chunking"]["strategy"],
                "chunk_size": c["chunking"]["chunk_size"],
                "overlap": c["chunking"]["overlap"],
                "documents": c["documents"],
                "chunks": result["indexing"]["chunks"],
                "embed_seconds": round(result["indexing"]["embed_seconds"], 3),
                "retrieval_available": result["retrieval"] is not None,
            }
        )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    import os

    import yaml

    from java_docs_rag.embeddings.factory import create_embedder
    from java_docs_rag.vectorstore.memory import InMemoryVectorStore
    from java_docs_rag.vectorstore.qdrant_store import QdrantVectorStore, collection_name_for

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--plan", type=Path, default=Path("configs/experiments/b_chunking_embeddings.yaml")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--group", default=None)
    parser.add_argument("--store", choices=["memory", "qdrant"], default="memory")
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=os.environ.get("EMBEDDING_DEVICE", "cpu"))
    parser.add_argument("--dry-run", action="store_true", help="print the run list only")
    args = parser.parse_args(argv)

    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    runs = expand_plan(plan, args.group)
    if args.dry_run:
        for run in runs:
            print(run)
        return 0
    documents = load_documents_jsonl(args.data_dir / "processed" / f"{plan['source_id']}.jsonl")
    if args.limit_docs:
        documents = documents[: args.limit_docs]

    embedders: dict[str, Embedder] = {}

    def make_embedder(model: str) -> Embedder:
        if model not in embedders:
            embedders[model] = create_embedder(
                EmbeddingConfig(model), batch_size=args.batch_size, device=args.device
            )
        return embedders[model]

    def make_store(model: str, tag: str) -> VectorStore:
        if args.store == "memory":
            return InMemoryVectorStore()
        store = QdrantVectorStore(
            VectorStoreConfig(
                url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
                collection=collection_name_for("exp_b", model, tag),
            )
        )
        store.delete_collection()  # each run starts from an empty collection
        return store

    for run in runs:
        print(f"running {run} on {len(documents)} documents ...", flush=True)
        result = run_one(
            run,
            documents,
            make_embedder,
            make_store,
            Path(plan["evaluation_dataset"]),
            plan["top_k"],
            args.batch_size,
        )
        print(f"  -> {save_result(result, args.results_dir)}")
        if result["retrieval"] is None:
            print(f"  retrieval metrics not computed: {result['retrieval_unavailable_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
