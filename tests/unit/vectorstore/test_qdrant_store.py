"""Developer B: Qdrant REST adapter against a fake HTTP transport."""

import json
from dataclasses import dataclass
from typing import Any

import pytest

from java_docs_rag.common.config import VectorStoreConfig
from java_docs_rag.common.exceptions import VectorStoreError
from java_docs_rag.common.models import DocumentChunk
from java_docs_rag.vectorstore.qdrant_store import (
    QdrantVectorStore,
    build_filter,
    collection_name_for,
    point_id,
)


@dataclass
class Response:
    status_code: int
    body: Any = None

    def json(self) -> Any:
        return self.body

    @property
    def text(self) -> str:
        return json.dumps(self.body)


class FakeQdrant:
    """Very small in-memory imitation of the Qdrant REST endpoints used by the adapter."""

    def __init__(self, unreachable: bool = False) -> None:
        self.collections: dict[str, dict[str, Any]] = {}
        self.requests: list[tuple[str, str, Any]] = []
        self.unreachable = unreachable

    def request(self, method: str, path: str, json: Any = None) -> Response:
        self.requests.append((method, path, json))
        if self.unreachable:
            raise ConnectionError("refused")
        parts = path.split("?")[0].strip("/").split("/")
        name = parts[1]
        coll = self.collections.get(name)
        if parts == ["collections", name] and method == "GET":
            if coll is None:
                return Response(404, {"status": {"error": "Not found"}})
            return Response(
                200, {"result": {"config": {"params": {"vectors": {"size": coll["size"]}}}}}
            )
        if parts == ["collections", name] and method == "PUT":
            self.collections[name] = {"size": json["vectors"]["size"], "points": {}}
            return Response(200, {"result": True})
        if parts == ["collections", name] and method == "DELETE":
            self.collections.pop(name, None)
            return Response(200, {"result": True})
        if coll is None:
            return Response(404, {"status": {"error": "Not found"}})
        action = "/".join(parts[2:])
        if action == "index":
            return Response(200, {"result": {}})
        if action == "points" and method == "PUT":
            for point in json["points"]:
                if len(point["vector"]) != coll["size"]:
                    return Response(400, {"status": {"error": "Wrong vector size"}})
                coll["points"][point["id"]] = point
            return Response(200, {"result": {}})
        if action == "points/search":

            def ok(payload: dict[str, Any]) -> bool:
                for cond in (json.get("filter") or {}).get("must", []):
                    value: Any = payload
                    for key in cond["key"].split("."):
                        value = value.get(key) if isinstance(value, dict) else None
                    if value != cond["match"]["value"]:
                        return False
                return True

            hits = [
                {
                    "id": pid,
                    "score": sum(a * b for a, b in zip(p["vector"], json["vector"], strict=True)),
                    "payload": p["payload"],
                }
                for pid, p in coll["points"].items()
                if ok(p["payload"])
            ]
            hits.sort(key=lambda h: -h["score"])
            return Response(200, {"result": hits[: json["limit"]]})
        if action == "points/delete":
            doc = json["filter"]["must"][0]["match"]["value"]
            coll["points"] = {
                k: v for k, v in coll["points"].items() if v["payload"]["document_id"] != doc
            }
            return Response(200, {"result": {}})
        if action == "points/count":
            return Response(200, {"result": {"count": len(coll["points"])}})
        raise AssertionError(f"unexpected request {method} {path}")


def chunk(doc: str, pos: int, package: str = "java.util") -> DocumentChunk:
    return DocumentChunk(
        f"{doc}#paragraph-s100-o0#{pos}",
        doc,
        f"text {doc} {pos}",
        f"https://x/{doc}",
        doc.title(),
        "Methods" if pos else None,
        pos,
        {"package": package, "java_version": "21"},
    )


def make_store(fake: FakeQdrant) -> QdrantVectorStore:
    return QdrantVectorStore(
        VectorStoreConfig("http://qdrant:6333", "test"),
        client=fake,
        extra_payload={"embedding_model": "fake"},
    )


def test_construction_has_no_side_effects() -> None:
    fake = FakeQdrant()
    make_store(fake)
    assert fake.requests == []
    assert QdrantVectorStore(VectorStoreConfig()).client is None


def test_upsert_creates_collection_and_search_round_trips_chunks() -> None:
    fake = FakeQdrant()
    store = make_store(fake)
    chunks = [chunk("list", 0), chunk("list", 1), chunk("map", 0, "java.util.concurrent")]
    store.upsert(chunks, [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    assert fake.collections["test"]["size"] == 2
    assert store.count() == 3
    stored = fake.collections["test"]["points"][point_id(chunks[1].chunk_id)]
    assert stored["payload"]["embedding_model"] == "fake"
    assert stored["payload"]["metadata"]["package"] == "java.util"

    results = store.search([1.0, 0.0], top_k=2)
    assert [r.chunk for r in results] == chunks[:2]
    assert results[0].similarity_score == pytest.approx(1.0)
    assert results[0].rerank_score is None
    filtered = store.search([1.0, 0.0], top_k=5, filters={"package": "java.util.concurrent"})
    assert [r.chunk.document_id for r in filtered] == ["map"]
    assert store.search([1.0, 0.0], top_k=0) == []


def test_reupsert_is_idempotent_and_delete_document() -> None:
    fake = FakeQdrant()
    store = make_store(fake)
    store.upsert([chunk("list", 0)], [[1.0, 0.0]])
    store.upsert([chunk("list", 0), chunk("map", 0)], [[1.0, 0.0], [0.0, 1.0]])
    assert store.count() == 2
    store.delete_document("list")
    assert [r.chunk.document_id for r in store.search([1.0, 0.0], 5)] == ["map"]


def test_batches_large_upserts() -> None:
    fake = FakeQdrant()
    store = make_store(fake)
    store.batch_size = 2
    store.upsert([chunk("d", i) for i in range(5)], [[1.0, 0.0]] * 5)
    puts = [r for r in fake.requests if r[0] == "PUT" and "/points" in r[1]]
    assert [len(r[2]["points"]) for r in puts] == [2, 2, 1]


def test_dimension_guards() -> None:
    fake = FakeQdrant()
    fake.collections["test"] = {"size": 3, "points": {}}
    store = make_store(fake)
    with pytest.raises(VectorStoreError, match="separate collection"):
        store.upsert([chunk("a", 0)], [[1.0, 0.0]])
    with pytest.raises(VectorStoreError, match="same dimension"):
        store.upsert([chunk("a", 0), chunk("a", 1)], [[1.0], [1.0, 0.0]])
    with pytest.raises(VectorStoreError, match="vectors"):
        store.upsert([chunk("a", 0)], [])


def test_errors_are_reported_clearly() -> None:
    with pytest.raises(VectorStoreError, match="does not exist"):
        make_store(FakeQdrant()).search([1.0], 3)
    with pytest.raises(VectorStoreError, match="unreachable"):
        make_store(FakeQdrant(unreachable=True)).search([1.0], 3)


def test_helpers() -> None:
    assert point_id("a") == point_id("a") != point_id("b")
    assert collection_name_for("java_docs", "BAAI/bge-small-en-v1.5", "paragraph-s1000-o0") == (
        "java_docs__bge-small-en-v1-5__paragraph-s1000-o0"
    )
    assert build_filter(None) is None
    assert build_filter({"document_id": "x", "package": "p"}) == {
        "must": [
            {"key": "document_id", "match": {"value": "x"}},
            {"key": "metadata.package", "match": {"value": "p"}},
        ]
    }
