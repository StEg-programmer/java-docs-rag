"""Developer B: experiment plan expansion and result format (offline, fake embedder)."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import yaml
from conftest import FakeEmbedder, make_document

from java_docs_rag.vectorstore.memory import InMemoryVectorStore

_SPEC = importlib.util.spec_from_file_location(
    "b_indexing_experiments", Path("experiments/runners/b_indexing_experiments.py")
)
assert _SPEC and _SPEC.loader
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner  # dataclasses need the module registered
_SPEC.loader.exec_module(runner)

PLAN = yaml.safe_load(Path("configs/experiments/b_chunking_embeddings.yaml").read_text())


def test_plan_changes_one_parameter_per_group() -> None:
    runs = runner.expand_plan(PLAN)
    by_group: dict[str, list] = {}
    for run in runs:
        by_group.setdefault(run.group, []).append(run)
    assert [r.strategy for r in by_group["chunking_strategy"]] == [
        "fixed",
        "overlap",
        "paragraph",
        "section",
    ]
    assert [r.overlap for r in by_group["chunking_strategy"]] == [0, 51, 0, 0]
    assert [r.chunk_size for r in by_group["chunk_size"]] == [256, 512, 1024]
    assert [r.overlap for r in by_group["overlap_percent"]] == [0, 51, 102]
    assert {r.model for r in by_group["chunk_size"]} == {"BAAI/bge-small-en-v1.5"}
    assert [r.model for r in by_group["embedding_model"]] == [
        "BAAI/bge-small-en-v1.5",
        "intfloat/multilingual-e5-small",
    ]
    assert runner.expand_plan(PLAN, "chunk_size") == by_group["chunk_size"]


def test_run_records_conditions_and_never_invents_retrieval_metrics(tmp_path: Path) -> None:
    documents = [
        make_document("list", "An ordered collection. " * 40),
        make_document("map", "Maps keys to values. " * 10),
    ]
    run = runner.RunConfig("chunk_size", "fake/letters", "paragraph", 256, 0)
    dataset = tmp_path / "questions.json"
    dataset.write_text("[]", encoding="utf-8")

    result = runner.run_one(
        run,
        documents,
        lambda _m: FakeEmbedder(),
        lambda _m, _t: InMemoryVectorStore(),
        dataset,
        [5, 10],
    )
    conditions = result["conditions"]
    assert conditions["chunking"] == {
        "strategy": "paragraph",
        "chunk_size": 256,
        "overlap": 0,
        "unit": "characters",
    }
    assert conditions["embedding_model"] == "fake/letters" and conditions["documents"] == 2
    assert conditions["evaluation_dataset_sha"] and len(conditions["corpus_version"]) == 16
    assert result["indexing"]["chunks"] > 2
    assert result["retrieval"] is None
    assert result["retrieval_unavailable_reason"] == (
        "evaluation dataset has no answerable questions with relevant documents"
    )

    missing = runner.run_one(
        run,
        documents,
        lambda _m: FakeEmbedder(),
        lambda _m, _t: InMemoryVectorStore(),
        tmp_path / "none.json",
        [5],
    )
    assert "not found" in missing["retrieval_unavailable_reason"]

    path = runner.save_result(result, tmp_path / "results")
    assert json.loads(path.read_text())["experiment_id"] == result["experiment_id"]
    runner.save_result(result, tmp_path / "results")
    rows = list(csv.DictReader((tmp_path / "results" / "b_indexing_summary.csv").open()))
    assert len(rows) == 2 and rows[0]["retrieval_available"] == "False"


def test_dry_run(capsys) -> None:
    assert runner.main(["--dry-run", "--group", "embedding_model"]) == 0
    assert capsys.readouterr().out.count("RunConfig") == 2


def test_real_a_metrics_apply_k_to_unique_documents(tmp_path: Path) -> None:
    from java_docs_rag.common.models import SearchResult
    from java_docs_rag.preprocessing.chunking.factory import create_chunker

    documents = [
        make_document("list", "Ordered collection. " * 30),
        make_document("map", "Maps keys to values."),
    ]
    chunker = create_chunker(runner.ChunkingConfig("paragraph", 100, 0))
    list_chunks = chunker.split(documents[0])
    map_chunk = chunker.split(documents[1])[0]
    hits = [
        SearchResult(list_chunks[0], 0.9),
        SearchResult(list_chunks[1], 0.8),
        SearchResult(map_chunk, 0.7),
    ]

    class Store:
        def search(self, vector, top_k):
            assert top_k == 3
            return hits

    dataset = tmp_path / "questions.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "map",
                    "question": "What is a Map?",
                    "expected_answer": None,
                    "relevant_document_ids": [documents[1].document_id],
                    "category": "api",
                    "answerable": True,
                },
                {
                    "question_id": "missing-labels",
                    "question": "No relevance labels",
                    "expected_answer": None,
                    "relevant_document_ids": [],
                    "category": "api",
                    "answerable": True,
                },
            ]
        ),
        encoding="utf-8",
    )
    metrics, reason = runner.evaluate_retrieval_quality(FakeEmbedder(), Store(), dataset, [2, 3])
    assert reason is None and metrics["questions"] == 1
    assert metrics["metrics_at_k"]["2"] == {
        "precision_at_k": 0.5,
        "recall_at_k": 1.0,
        "mrr_at_k": 0.5,
    }
