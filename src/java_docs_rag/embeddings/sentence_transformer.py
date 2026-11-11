"""Developer B: local Sentence Transformers embedding adapter (implements ``Embedder``).

``sentence_transformers`` is imported and the model is downloaded/loaded only on the
first call that needs it, so importing and constructing the class is free of I/O.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from java_docs_rag.common.exceptions import EmbeddingError


@dataclass
class EmbeddingStats:
    calls: int = 0
    texts: int = 0
    seconds: float = 0.0
    last_seconds: float = 0.0

    @property
    def texts_per_second(self) -> float:
        return self.texts / self.seconds if self.seconds > 0 else 0.0


def _load_sentence_transformer(model_id: str, device: str, cache_folder: str | None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise EmbeddingError(
            'sentence-transformers is not installed; run: pip install -e ".[embeddings]"'
        ) from exc
    return SentenceTransformer(model_id, device=device, cache_folder=cache_folder)


class SentenceTransformerEmbedder:
    """Embeds passages and queries with optional model-specific prefixes.

    E5 models require ``"query: "`` / ``"passage: "`` prefixes; BGE v1.5 recommends an
    instruction for short retrieval queries and no prefix for passages.
    """

    def __init__(
        self,
        model_id: str,
        dimension: int | None = None,
        *,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        cache_folder: str | None = None,
        model_loader: Callable[[str, str, str | None], Any] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._model_id = model_id
        self._expected_dimension = dimension
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.cache_folder = cache_folder
        self._loader = model_loader or _load_sentence_transformer
        self._model: Any = None
        self.stats = EmbeddingStats()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        if self._expected_dimension is not None:
            return self._expected_dimension
        model = self._get_model()
        size = model.get_sentence_embedding_dimension()
        if not size:
            raise EmbeddingError(f"Model {self._model_id} does not report its dimension")
        self._expected_dimension = int(size)
        return self._expected_dimension

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                self._model = self._loader(self._model_id, self.device, self.cache_folder)
            except EmbeddingError:
                raise
            except Exception as exc:  # model hub / IO errors are wrapped for callers
                raise EmbeddingError(
                    f"Cannot load embedding model {self._model_id}: {exc}"
                ) from exc
        return self._model

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        started = time.perf_counter()
        try:
            raw = model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding with {self._model_id} failed: {exc}") from exc
        elapsed = time.perf_counter() - started
        vectors = [
            [float(x) for x in row] for row in (raw.tolist() if hasattr(raw, "tolist") else raw)
        ]
        if len(vectors) != len(texts):
            raise EmbeddingError(f"Expected {len(texts)} vectors, got {len(vectors)}")
        expected = self.dimension
        bad = next((len(v) for v in vectors if len(v) != expected), None)
        if bad is not None:
            raise EmbeddingError(
                f"Model {self._model_id} returned dimension {bad}, expected {expected}"
            )
        self.stats.calls += 1
        self.stats.texts += len(texts)
        self.stats.seconds += elapsed
        self.stats.last_seconds = elapsed
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([self.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]
