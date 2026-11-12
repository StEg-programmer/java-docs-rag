"""Lazy scalar cross-encoder scorer for the existing injected reranker contract."""

from collections.abc import Callable, Sequence
from math import isfinite
from typing import Any

from java_docs_rag.common.exceptions import ConfigurationError, JavaDocsRAGError

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _load_model(model_id: str, device: str) -> Any:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ConfigurationError(
            'Reranking requires the existing optional extra: pip install -e ".[embeddings]"'
        ) from exc
    # Provision models separately; scoring must not silently download weights.
    return CrossEncoder(model_id, device=device, local_files_only=True)


class SentenceTransformerScorer:
    """Load a locally available single-score model on first nonempty scoring call."""

    def __init__(
        self,
        model_id: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        model_loader: Callable[[str, str], Any] = _load_model,
    ) -> None:
        if not model_id.strip():
            raise ValueError("reranker model_id must not be blank")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._loader = model_loader
        self._model: Any = None

    def __call__(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        try:
            if self._model is None:
                self._model = self._loader(self.model_id, self.device)
            raw = self._model.predict(
                list(pairs),
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            values = raw.tolist() if hasattr(raw, "tolist") else raw
            scores = [float(value) for value in values]
            if len(scores) != len(pairs) or not all(isfinite(score) for score in scores):
                raise ValueError("expected one finite scalar score per query/text pair")
            return scores
        except JavaDocsRAGError:
            raise
        except Exception as exc:
            raise JavaDocsRAGError(f"Reranker {self.model_id} failed: {exc}") from exc
