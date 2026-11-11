"""Developer B: Qdrant adapter over the REST API; construction has no network side effects.

The adapter talks to Qdrant through a small HTTP client (``httpx.Client`` by default,
created on first use), which keeps ``qdrant-client`` out of the dependency set and
lets unit tests replace the transport with a fake.

Schema (docs/decisions/0001-indexing-identity-and-chunking.md):

* one collection per (embedding model, chunking configuration) — never mix models;
* point ID = UUIDv5(chunk_id), so re-upserting a chunk overwrites it;
* payload = all DocumentChunk fields, ``metadata`` nested, plus ``extra_payload``;
* distance = Cosine; ``similarity_score`` is Qdrant's score (higher is closer).
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from java_docs_rag.common.config import VectorStoreConfig
from java_docs_rag.common.exceptions import VectorStoreError
from java_docs_rag.common.models import DocumentChunk, SearchResult

CHUNK_FIELDS = ("chunk_id", "document_id", "source_url", "title", "section", "position")
_KEYWORD_INDEXES = ("document_id", "metadata.java_version", "metadata.package")
_NAMESPACE = uuid.UUID("6f0b6c1e-2a4d-5d7a-9a52-4b1f0f6f7a01")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def collection_name_for(base: str, model_id: str, chunking_tag: str | None = None) -> str:
    """``java_docs`` + ``BAAI/bge-small-en-v1.5`` + ``paragraph-s1000-o0`` →
    ``java_docs__bge-small-en-v1-5__paragraph-s1000-o0``."""
    model = re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")
    parts = [base, model] + ([chunking_tag.lower()] if chunking_tag else [])
    return "__".join(parts)


def _payload(chunk: DocumentChunk, extra: dict[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {name: getattr(chunk, name) for name in CHUNK_FIELDS}
    payload["text"] = chunk.text
    payload["metadata"] = dict(chunk.metadata)
    payload.update(extra)
    return payload


def _chunk(payload: dict[str, Any]) -> DocumentChunk:
    try:
        return DocumentChunk(
            chunk_id=str(payload["chunk_id"]),
            document_id=str(payload["document_id"]),
            text=str(payload["text"]),
            source_url=str(payload["source_url"]),
            title=str(payload["title"]),
            section=payload.get("section"),
            position=int(payload["position"]),
            metadata={str(k): str(v) for k, v in (payload.get("metadata") or {}).items()},
        )
    except KeyError as exc:
        raise VectorStoreError(f"Qdrant point payload is missing {exc}") from exc


def build_filter(filters: dict[str, str] | None) -> dict[str, Any] | None:
    """Exact-match filter; chunk fields are top-level, other keys refer to metadata."""
    if not filters:
        return None
    must = []
    for key, value in sorted(filters.items()):
        path = key if key in CHUNK_FIELDS or key.startswith("metadata.") else f"metadata.{key}"
        must.append({"key": path, "match": {"value": value}})
    return {"must": must}


@dataclass
class QdrantVectorStore:
    config: VectorStoreConfig
    client: Any = None
    api_key: str | None = None
    timeout: float = 30.0
    batch_size: int = 128
    distance: str = "Cosine"
    extra_payload: dict[str, str] = field(default_factory=dict)
    _dimension: int | None = field(default=None, init=False, repr=False)

    @property
    def collection(self) -> str:
        return self.config.collection

    # -- transport ----------------------------------------------------------------

    def _http(self) -> Any:
        if self.client is None:
            import httpx

            headers = {"api-key": self.api_key} if self.api_key else {}
            self.client = httpx.Client(
                base_url=self.config.url.rstrip("/"), headers=headers, timeout=self.timeout
            )
        return self.client

    def _call(
        self, method: str, path: str, body: dict[str, Any] | None = None, allow_404: bool = False
    ) -> Any:
        try:
            response = self._http().request(method, path, json=body)
        except Exception as exc:
            raise VectorStoreError(
                f"Qdrant at {self.config.url} is unreachable ({type(exc).__name__}: {exc}). "
                "Is the container running? See docs/setup.md."
            ) from exc
        if response.status_code == 404 and allow_404:
            return None
        if response.status_code >= 400:
            raise VectorStoreError(
                f"Qdrant {method} {path} failed: HTTP {response.status_code} {response.text[:300]}"
            )
        data = response.json()
        return data.get("result") if isinstance(data, dict) else data

    # -- collection management ------------------------------------------------------

    def collection_dimension(self) -> int | None:
        """Vector size of the existing collection, or None if it does not exist."""
        info = self._call("GET", f"/collections/{self.collection}", allow_404=True)
        if info is None:
            return None
        vectors = info["config"]["params"]["vectors"]
        return int(vectors["size"])

    def ensure_collection(self, dimension: int) -> None:
        if self._dimension == dimension:
            return
        existing = self.collection_dimension()
        if existing is None:
            self._call(
                "PUT",
                f"/collections/{self.collection}",
                {"vectors": {"size": dimension, "distance": self.distance}},
            )
            for name in _KEYWORD_INDEXES:
                self._call(
                    "PUT",
                    f"/collections/{self.collection}/index?wait=true",
                    {"field_name": name, "field_schema": "keyword"},
                )
        elif existing != dimension:
            raise VectorStoreError(
                f"Collection '{self.collection}' stores {existing}-d vectors, got {dimension}-d. "
                "Use a separate collection per embedding model."
            )
        self._dimension = dimension

    def delete_collection(self) -> None:
        self._call("DELETE", f"/collections/{self.collection}", allow_404=True)
        self._dimension = None

    def count(self) -> int:
        result = self._call(
            "POST",
            f"/collections/{self.collection}/points/count",
            {"exact": True},
            allow_404=True,
        )
        return int(result["count"]) if result else 0

    # -- VectorStore protocol -------------------------------------------------------

    def upsert(self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return
        dimension = len(vectors[0])
        if any(len(v) != dimension for v in vectors):
            raise VectorStoreError("All vectors in one upsert must have the same dimension")
        self.ensure_collection(dimension)
        for start in range(0, len(chunks), self.batch_size):
            points = [
                {
                    "id": point_id(chunk.chunk_id),
                    "vector": [float(x) for x in vector],
                    "payload": _payload(chunk, self.extra_payload),
                }
                for chunk, vector in zip(
                    chunks[start : start + self.batch_size],
                    vectors[start : start + self.batch_size],
                    strict=True,
                )
            ]
            self._call(
                "PUT", f"/collections/{self.collection}/points?wait=true", {"points": points}
            )

    def search(
        self, vector: Sequence[float], top_k: int, filters: dict[str, str] | None = None
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        body: dict[str, Any] = {
            "vector": [float(x) for x in vector],
            "limit": top_k,
            "with_payload": True,
        }
        query_filter = build_filter(filters)
        if query_filter:
            body["filter"] = query_filter
        result = self._call(
            "POST", f"/collections/{self.collection}/points/search", body, allow_404=True
        )
        if result is None:
            raise VectorStoreError(
                f"Collection '{self.collection}' does not exist; run indexing first"
            )
        return [
            SearchResult(chunk=_chunk(point["payload"]), similarity_score=float(point["score"]))
            for point in result
        ]

    def delete_document(self, document_id: str) -> None:
        self._call(
            "POST",
            f"/collections/{self.collection}/points/delete?wait=true",
            {"filter": build_filter({"document_id": document_id})},
            allow_404=True,
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
