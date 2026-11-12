"""Publish verified Qdrant builds through aliases without mutating the old collection.

This administrative composition layer does not change B's VectorStore Protocol or
IndexingPipeline. Use a single writer per alias; retain old/failed builds for review.
"""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from math import isclose, isfinite
from typing import Any
from uuid import uuid4

from java_docs_rag.common.config import VectorStoreConfig
from java_docs_rag.common.exceptions import VectorStoreError
from java_docs_rag.common.interfaces import Chunker, Embedder
from java_docs_rag.common.models import Document, DocumentChunk
from java_docs_rag.observability.safe_tracing import SafeTelemetry
from java_docs_rag.pipeline.indexing_pipeline import IndexingPipeline, IndexingReport
from java_docs_rag.vectorstore.qdrant_store import QdrantVectorStore, point_id


@dataclass
class QdrantCollectionAdmin:
    config: VectorStoreConfig
    api_key: str | None = None
    client: Any = None
    timeout: float = 30.0

    def call(self, method: str, path: str, body: dict | None = None) -> Any:
        """Administrative REST boundary, created lazily and injectable for offline tests."""
        if self.client is None:
            import httpx

            self.client = httpx.Client(
                base_url=self.config.url.rstrip("/"),
                timeout=self.timeout,
                headers={"api-key": self.api_key} if self.api_key else {},
            )
        try:
            response = self.client.request(method, path, json=body)
            if response.status_code == 404 and method == "GET":
                return None
            if response.status_code >= 400:
                raise VectorStoreError(f"Qdrant admin {method} {path}: HTTP {response.status_code}")
            data = response.json()
            if not isinstance(data, dict) or "result" not in data:
                raise ValueError("missing result")
            return data["result"]
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Qdrant admin {method} {path} failed: {exc}") from exc

    def aliases(self) -> dict[str, str]:
        result = self.call("GET", "/aliases")
        if not isinstance(result, dict) or not isinstance(result.get("aliases"), list):
            raise VectorStoreError("Qdrant alias listing is invalid")
        return {entry["alias_name"]: entry["collection_name"] for entry in result["aliases"]}

    def validate_collection(self, name: str, dimension: int) -> bool:
        """Read-only check for an existing unnamed Cosine collection of the right size."""
        info = self.call("GET", f"/collections/{name}")
        if info is None:
            return False
        try:
            vectors = info["config"]["params"]["vectors"]
            compatible = vectors["size"] == dimension and vectors["distance"] == "Cosine"
        except (KeyError, TypeError):
            compatible = False
        if not compatible:
            raise VectorStoreError(
                f"Collection {name!r} must use unnamed {dimension}-d Cosine vectors"
            )
        return True

    def points(self, name: str) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        offset = None
        seen_offsets: set[str] = set()
        while True:
            body = {"limit": 128, "with_payload": True, "with_vector": True}
            if offset is not None:
                body["offset"] = offset
            page = self.call("POST", f"/collections/{name}/points/scroll", body)
            if not isinstance(page, dict) or not isinstance(page.get("points"), list):
                raise VectorStoreError(f"Invalid scroll response for {name!r}")
            points.extend(page["points"])
            offset = page.get("next_page_offset")
            if offset is None:
                return points
            if str(offset) in seen_offsets:
                raise VectorStoreError("Qdrant scroll repeated its offset")
            seen_offsets.add(str(offset))

    def copy_points(self, name: str, points: Sequence[dict[str, Any]]) -> None:
        for start in range(0, len(points), 128):
            batch = [
                {key: point[key] for key in ("id", "vector", "payload")}
                for point in points[start : start + 128]
            ]
            self.call("PUT", f"/collections/{name}/points?wait=true", {"points": batch})

    def publish(self, alias: str, build: str, previous: str | None) -> None:
        """Switch both alias actions atomically; never delete any physical collection."""
        if self.aliases().get(alias) != previous:
            raise VectorStoreError(
                "Published alias changed during rebuild; refusing to overwrite it"
            )
        actions = []
        if previous is not None:
            actions.append({"delete_alias": {"alias_name": alias}})
        actions.append({"create_alias": {"alias_name": alias, "collection_name": build}})
        # A timeout here has an uncertain outcome: retain both physical collections.
        self.call("POST", "/collections/aliases", {"actions": actions})

    def close(self) -> None:
        if self.client is not None:
            self.client.close()


def _validate_points(points: Sequence[dict], dimension: int, model: str, chunking: str) -> None:
    seen = set()
    for point in points:
        try:
            vector, payload = point["vector"], point["payload"]
            valid = (
                point["id"] not in seen
                and isinstance(vector, list)
                and len(vector) == dimension
                and all(isinstance(value, (int, float)) and isfinite(value) for value in vector)
                and payload["embedding_model"] == model
                and payload["chunking"] == chunking
                and isinstance(payload["document_id"], str)
                and point_id(payload["chunk_id"]) == point["id"]
            )
        except (KeyError, TypeError, AttributeError):
            valid = False
        if not valid:
            raise VectorStoreError("Collection points have incompatible vectors/model/chunking/IDs")
        seen.add(point["id"])


@dataclass(frozen=True)
class _PreparedChunker:
    config_tag: str
    chunks_by_document: dict[str, list[DocumentChunk]]

    def split(self, document: Document) -> list[DocumentChunk]:
        """Return the exact chunks validated before any collection mutation."""
        return list(self.chunks_by_document[document.document_id])


def rebuild_collection(
    target: str,
    documents: Sequence[Document],
    chunker: Chunker,
    embedder: Embedder,
    *,
    admin: QdrantCollectionAdmin,
    store_factory: Callable[[str], QdrantVectorStore],
    chunking_tag: str,
    batch_size: int = 64,
    from_collection: str | None = None,
    tracer: SafeTelemetry | None = None,
) -> tuple[IndexingReport, str, str | None]:
    """Stage a subset replacement, verify all points, then atomically publish its alias.

    Existing physical targets are refused: migrate via a NEW alias and
    from_collection instead. No previous collection is deleted on success/failure.
    The source must be quiescent; snapshots/alias updates do not provide a writer lock.
    """
    if not documents or len({doc.document_id for doc in documents}) != len(documents):
        raise ValueError("index requires nonempty documents with unique document IDs")
    prepared: dict[str, list[DocumentChunk]] = {}
    for document in documents:
        document_chunks = [chunk for chunk in chunker.split(document) if chunk.text.strip()]
        if not document_chunks:
            raise ValueError(
                f"Replacement document {document.document_id!r} produced no indexable chunks"
            )
        prepared[document.document_id] = document_chunks
    prepared_chunker = _PreparedChunker(chunking_tag, prepared)
    chunks = [chunk for document_chunks in prepared.values() for chunk in document_chunks]
    dimension = embedder.dimension
    previous = admin.aliases().get(target)
    exists = admin.validate_collection(target, dimension)
    if exists and previous is None:
        raise VectorStoreError(
            f"{target!r} is a physical collection; cannot atomically replace it. "
            "Use --collection NEW_ALIAS --from-collection EXISTING_COLLECTION."
        )
    if previous is not None and not exists:
        raise VectorStoreError("Published alias has no readable collection")
    if previous and from_collection:
        raise ValueError("--from-collection is only allowed for a new target alias")
    source = previous or from_collection
    original = []
    if source:
        if not admin.validate_collection(source, dimension):
            raise VectorStoreError(f"Source collection {source!r} does not exist")
        original = admin.points(source)
        _validate_points(original, dimension, embedder.model_id, chunking_tag)
    replaced_ids = {doc.document_id for doc in documents}
    retained = [point for point in original if point["payload"]["document_id"] not in replaced_ids]
    expected_payloads = {point["id"]: point["payload"] for point in retained}
    for chunk in chunks:
        expected_payloads[point_id(chunk.chunk_id)] = {
            **asdict(chunk),
            "embedding_model": embedder.model_id,
            "chunking": chunking_tag,
        }
    build = f"{target}__build_{uuid4().hex}"
    stage = store_factory(build)
    try:
        stage.ensure_collection(dimension)
        admin.copy_points(build, retained)
        # B indexes into an unpublished, isolated store. No fake previous hashes or deletions.
        report = IndexingPipeline(
            prepared_chunker,
            embedder,
            stage,
            batch_size,
            tracer=tracer if tracer is not None else SafeTelemetry(),
        ).run(documents)
        written = admin.points(build)
        _validate_points(written, dimension, embedder.model_id, chunking_tag)
        actual = {point["id"]: point["payload"] for point in written}
        if actual != expected_payloads:
            raise VectorStoreError("Staged build verification failed: point IDs/payloads differ")
        copied_vectors = {point["id"]: point["vector"] for point in written}
        for point in retained:
            if not all(
                isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
                for a, b in zip(point["vector"], copied_vectors[point["id"]], strict=True)
            ):
                raise VectorStoreError("Staged build verification failed: retained vector changed")
        admin.publish(target, build, previous)
        return report, build, previous
    except Exception as exc:
        raise VectorStoreError(
            f"Rebuild {build!r} failed: {exc}. Old physical collection {source!r} is retained. "
            "If alias publication timed out, inspect /aliases before retrying."
        ) from exc
    finally:
        stage.close()
