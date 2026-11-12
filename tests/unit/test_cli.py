"""Supported CLI wiring; storage and model boundaries are deterministic and offline."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
from test_collection_rebuild import FakeQdrant
from test_offline_integration import (
    DOCUMENT_ID,
    FIXTURE,
    SOURCE_URL,
    DeterministicEmbeddingModel,
    DeterministicScoringModel,
    EvidenceProvider,
)
from test_safe_tracing import RecordingTracer

from java_docs_rag import cli
from java_docs_rag.common.config import VectorStoreConfig
from java_docs_rag.embeddings.factory import create_embedder
from java_docs_rag.grabber.identity import content_hash
from java_docs_rag.pipeline import collection_rebuild
from java_docs_rag.pipeline.indexing_pipeline import DocumentPreparer, save_documents_jsonl
from java_docs_rag.reranking.sentence_transformer import SentenceTransformerScorer
from java_docs_rag.vectorstore.qdrant_store import QdrantVectorStore


@pytest.fixture
def runtime(monkeypatch):
    model, scoring_model = DeterministicEmbeddingModel(), DeterministicScoringModel()
    provider, qdrant = EvidenceProvider(), FakeQdrant()

    def factory(config, **kwargs):
        embedder = create_embedder(config, **kwargs)
        embedder._loader = lambda *_: model
        return embedder

    monkeypatch.setattr(cli, "create_embedder", factory)
    monkeypatch.setattr(
        cli,
        "QdrantVectorStore",
        lambda config, **kwargs: QdrantVectorStore(config, client=qdrant, **kwargs),
    )
    admin_class = collection_rebuild.QdrantCollectionAdmin
    monkeypatch.setattr(
        collection_rebuild,
        "QdrantCollectionAdmin",
        lambda config, **kwargs: admin_class(config, client=qdrant, **kwargs),
    )
    monkeypatch.setattr(cli, "OllamaProvider", lambda *_: provider)
    monkeypatch.setattr(
        cli,
        "SentenceTransformerScorer",
        lambda name, **kwargs: SentenceTransformerScorer(
            name, model_loader=lambda *_: scoring_model, **kwargs
        ),
    )
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    return qdrant, provider


def test_index_and_ask_use_same_default_model_dimension_collection(runtime, capsys) -> None:
    assert cli.main(["index", "--html", str(FIXTURE), "--source-url", SOURCE_URL]) == 0
    indexed = json.loads(capsys.readouterr().out)
    assert indexed["chunks"] > 0
    assert cli.main(["ask", "What does List.size() return?", "--ollama-model", "fake"]) == 0
    answer = json.loads(capsys.readouterr().out)
    for field in ("collection", "embedding_model", "dimension"):
        assert indexed[field] == answer[field]
    assert answer["collection"] == "java_docs__bge-small-en-v1-5__paragraph-s1000-o0"
    assert answer["dimension"] == 384 and not answer["insufficient_context"]
    assert answer["sources"][0]["document_id"] == DOCUMENT_ID
    assert answer["sources"][0]["source_url"] == SOURCE_URL
    assert (
        cli.main(
            ["ask", "Tomorrow's weather?", "--ollama-model", "fake", "--min-similarity", "0.5"]
        )
        == 0
    )
    outside = json.loads(capsys.readouterr().out)
    assert outside["insufficient_context"] and outside["sources"] == []
    assert len(runtime[1].calls) == 1


def test_canonical_alias_and_explicit_collection_match_b_naming(runtime) -> None:
    parser = cli._parser()
    index = parser.parse_args(["index", "--input", "x", "--model", "bge-small"])
    ask = parser.parse_args(["ask", "q", "--model", "BAAI/bge-small-en-v1.5"])
    assert cli._resources(index)[2].collection == cli._resources(ask)[2].collection
    args = parser.parse_args(["ask", "q", "--collection", "isolated-smoke"])
    assert cli._resources(args)[2].collection == "isolated-smoke"


def test_ask_missing_or_wrong_collection_is_error(runtime, monkeypatch, capsys) -> None:
    assert cli.main(["ask", "q", "--ollama-model", "fake"]) == 1
    assert "dimension is None" in capsys.readouterr().err
    runtime[0].collections["java_docs__bge-small-en-v1-5__paragraph-s1000-o0"] = {
        "size": 768,
        "distance": "Cosine",
        "points": {},
    }
    assert cli.main(["ask", "q", "--ollama-model", "fake"]) == 1
    assert "expected 384" in capsys.readouterr().err


def test_required_inputs_fail_before_creating_resources(runtime) -> None:
    for args in (["index", "--html", str(FIXTURE)], ["ask", "q"]):
        with pytest.raises(SystemExit) as caught:
            cli.main(args)
        assert caught.value.code == 2
    assert runtime[0].collections == {}


def test_existing_b_indexing_cli_is_forwarded(monkeypatch) -> None:
    from java_docs_rag.pipeline import indexing_pipeline

    calls = []
    monkeypatch.setattr(indexing_pipeline, "main", lambda args: calls.append(args) or 0)
    assert cli.main(["indexing", "--data-dir", "data", "prepare"]) == 0
    assert calls == [["--data-dir", "data", "prepare"]]


def _index_args():
    return ["index", "--html", str(FIXTURE), "--source-url", SOURCE_URL]


@pytest.mark.parametrize(
    "failure", ["model", "dimension", "embedding-dimension", "write", "verify", "publish"]
)
def test_failed_replacement_keeps_original_searchable(
    runtime, monkeypatch, capsys, failure
) -> None:
    qdrant, _ = runtime
    assert cli.main(_index_args()) == 0
    initial = json.loads(capsys.readouterr().out)
    target, original_build = initial["collection"], initial["build_collection"]
    original = deepcopy(qdrant.collections[original_build])
    store = QdrantVectorStore(VectorStoreConfig(collection=target), client=qdrant)
    original_hits = store.search([1.0] + [0.0] * 383, 20)
    assert original_hits and original_hits[0].chunk.document_id == DOCUMENT_ID
    before = len(qdrant.requests)
    args = _index_args()
    if failure in {"model", "embedding-dimension"}:

        def factory(config, **kwargs):
            embedder = create_embedder(config, **kwargs)

            def broken(*_):
                if failure == "embedding-dimension":

                    class WrongDimensionModel:
                        def encode(self, texts, **kwargs):
                            return [[1.0, 0.0] for _ in texts]

                    return WrongDimensionModel()
                raise OSError("forced model load failure")

            embedder._loader = broken
            return embedder

        monkeypatch.setattr(cli, "create_embedder", factory)
    elif failure == "dimension":
        args += ["--model", "custom/model", "--dimension", "768", "--collection", target]
    elif failure == "write":
        qdrant.fail_write = True
    elif failure == "verify":
        qdrant.drop_write = True
    else:
        qdrant.fail_publish = True
    assert cli.main(args) == 1
    assert capsys.readouterr().err
    assert qdrant.aliases[target] == original_build
    assert qdrant.collections[original_build] == original
    assert store.search([1.0] + [0.0] * 383, 20) == original_hits
    if failure == "dimension":
        assert all(method == "GET" for method, _, _ in qdrant.requests[before:-1])
    assert not any(
        "points/delete" in path or method == "DELETE" for method, path, _ in qdrant.requests
    )


def test_verified_replacement_switches_alias_and_keeps_other_documents(runtime, tmp_path, capsys):
    qdrant, _ = runtime
    document = DocumentPreparer().prepare(FIXTURE.read_bytes(), SOURCE_URL)
    other = replace(document, document_id="other", source_url="https://example.test/other")
    path = tmp_path / "input.jsonl"
    save_documents_jsonl([document, other], path)
    assert cli.main(["index", "--input", str(path)]) == 0
    initial = json.loads(capsys.readouterr().out)
    old = deepcopy(qdrant.collections[initial["build_collection"]])
    shorter = replace(
        document,
        content="A list stores elements.",
        content_hash=content_hash("A list stores elements."),
    )
    save_documents_jsonl([shorter], path)
    assert cli.main(["index", "--input", str(path)]) == 0
    updated = json.loads(capsys.readouterr().out)
    assert updated["previous_collection"] == initial["build_collection"]
    assert qdrant.aliases[updated["collection"]] == updated["build_collection"]
    assert qdrant.collections[initial["build_collection"]] == old
    current = qdrant.collections[updated["build_collection"]]["points"]
    assert (
        len([point for point in current.values() if point["payload"]["document_id"] == DOCUMENT_ID])
        == 1
    )
    retained = {
        key: point
        for key, point in old["points"].items()
        if point["payload"]["document_id"] == "other"
    }
    assert retained and all(current[key] == point for key, point in retained.items())
    assert updated["chunks"] == 1


@pytest.mark.parametrize(("field", "value"), [("distance", "Euclid"), ("size", 768)])
def test_incompatible_collection_is_rejected_before_mutation(runtime, capsys, field, value):
    qdrant, _ = runtime
    assert cli.main(_index_args()) == 0
    initial = json.loads(capsys.readouterr().out)
    qdrant.collections[initial["build_collection"]][field] = value
    before = len(qdrant.requests)
    assert cli.main(_index_args()) == 1
    assert "Cosine vectors" in capsys.readouterr().err
    assert all(method == "GET" for method, _, _ in qdrant.requests[before:])


def test_physical_target_requires_explicit_nondestructive_migration(runtime, capsys):
    qdrant, _ = runtime
    assert cli.main(_index_args()) == 0
    initial = json.loads(capsys.readouterr().out)
    physical = initial["build_collection"]
    original = deepcopy(qdrant.collections[physical])
    before = len(qdrant.requests)
    assert cli.main(_index_args() + ["--collection", physical]) == 1
    assert "--from-collection" in capsys.readouterr().err
    assert all(method == "GET" for method, _, _ in qdrant.requests[before:])
    assert (
        cli.main(_index_args() + ["--collection", "migrated", "--from-collection", physical]) == 0
    )
    migrated = json.loads(capsys.readouterr().out)
    assert qdrant.aliases["migrated"] == migrated["build_collection"]
    assert qdrant.collections[physical] == original


@pytest.mark.parametrize("field", ["embedding_model", "chunking"])
def test_same_dimension_incompatible_payload_is_rejected_before_mutation(runtime, capsys, field):
    qdrant, _ = runtime
    assert cli.main(_index_args()) == 0
    initial = json.loads(capsys.readouterr().out)
    first = next(iter(qdrant.collections[initial["build_collection"]]["points"].values()))
    first["payload"][field] = "incompatible"
    original = deepcopy(qdrant.collections)
    before = len(qdrant.requests)
    assert cli.main(_index_args()) == 1
    assert "incompatible" in capsys.readouterr().err
    assert qdrant.collections == original
    assert all(
        method == "GET" or path.endswith("/scroll") for method, path, _ in qdrant.requests[before:]
    )


@pytest.mark.parametrize("empty_content", ["", " \t\n"])
@pytest.mark.parametrize("empty_first", [False, True])
def test_batch_with_empty_replacement_creates_no_stage_and_keeps_original(
    runtime, tmp_path, capsys, empty_content, empty_first
) -> None:
    qdrant, _ = runtime
    assert cli.main(_index_args()) == 0
    initial = json.loads(capsys.readouterr().out)
    target = initial["collection"]
    store = QdrantVectorStore(VectorStoreConfig(collection=target), client=qdrant)
    original_hits = store.search([1.0] + [0.0] * 383, 20)
    assert original_hits and original_hits[0].chunk.document_id == DOCUMENT_ID
    document = DocumentPreparer().prepare(FIXTURE.read_bytes(), SOURCE_URL)
    empty = replace(document, content=empty_content, content_hash=content_hash(empty_content))
    valid = replace(document, document_id="another-valid-document")
    path = tmp_path / "mixed.jsonl"
    save_documents_jsonl([empty, valid] if empty_first else [valid, empty], path)
    original_collections, original_aliases = deepcopy(qdrant.collections), dict(qdrant.aliases)
    before = len(qdrant.requests)

    assert cli.main(["index", "--input", str(path)]) == 1

    assert f"{DOCUMENT_ID!r} produced no indexable chunks" in capsys.readouterr().err
    assert qdrant.requests[before:] == []  # Batch rejected before even administrative reads.
    assert qdrant.collections == original_collections  # No staging collection was created.
    assert qdrant.aliases == original_aliases
    assert store.search([1.0] + [0.0] * 383, 20) == original_hits


def test_trace_flag_wires_index_and_ask_with_count_only_metadata(runtime, monkeypatch, capsys):
    backend = RecordingTracer()
    monkeypatch.setattr(cli, "create_tracer", lambda config: backend)
    for name in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.setenv(name, "credential-SENSITIVE")
    assert cli.main(_index_args() + ["--trace"]) == 0
    indexed = json.loads(capsys.readouterr().out)
    assert indexed["telemetry"]["trace_ids"] == ["a" * 32]
    assert [event["name"] for event in backend.events] == [
        "index.command",
        "indexing.run",
        "indexing.embed_and_upsert",
    ]
    assert backend.events[0]["output"]["published"]
    assert backend.events[1]["output"]["documents_indexed"] == 1
    assert all(event["output"]["duration_ms"] >= 0 for event in backend.events)
    backend.events.clear()
    assert cli.main(["ask", "question-SENSITIVE", "--ollama-model", "fake", "--trace"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["telemetry"]["receipt_verified"] is False
    assert backend.events[0]["name"] == "ask.command"
    assert backend.events[0]["output"]["raw_count"] > 0
    assert "SENSITIVE" not in repr(backend.events)
    assert "number of elements" not in repr(backend.events)


def test_trace_requires_local_environment_before_any_storage_work(runtime, monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with pytest.raises(SystemExit) as caught:
        cli.main(_index_args() + ["--trace"])
    assert caught.value.code == 2
    assert runtime[0].requests == []


def test_cli_continues_when_telemetry_is_unavailable(runtime, monkeypatch, capsys):
    for name in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.setenv(name, "credential-SENSITIVE")
    monkeypatch.setattr(cli, "create_tracer", lambda config: RecordingTracer("start"))
    assert cli.main(_index_args() + ["--trace"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["chunks"] > 0 and output["telemetry"]["errors"] > 0
