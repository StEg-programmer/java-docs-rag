"""The real scoring adapter with a fake model loader, no imports of ML libraries."""

import pytest

from java_docs_rag.common.exceptions import JavaDocsRAGError
from java_docs_rag.reranking.sentence_transformer import SentenceTransformerScorer


def test_lazy_batch_scoring_loads_once_and_preserves_pairs() -> None:
    loads, calls = [], []

    class Model:
        def predict(self, pairs, **kwargs):
            calls.append((pairs, kwargs))
            return [0.2, -0.5]

    def loader(model_id, device):
        loads.append((model_id, device))
        return Model()

    scorer = SentenceTransformerScorer("local", batch_size=7, model_loader=loader)
    assert loads == []
    assert scorer([]) == []
    pairs = [("q", "first"), ("q", "second")]
    assert scorer(pairs) == [0.2, -0.5]
    assert scorer(pairs) == [0.2, -0.5]
    assert loads == [("local", "cpu")]
    assert calls[0] == (
        pairs,
        {
            "batch_size": 7,
            "show_progress_bar": False,
            "convert_to_numpy": True,
        },
    )


@pytest.mark.parametrize("values", [[0.2], [[0.1, 0.9], [0.8, 0.2]], [float("nan"), 0.1]])
def test_invalid_scalar_model_scores_are_clear_errors(values) -> None:
    class Model:
        def predict(self, *args, **kwargs):
            return values

    scorer = SentenceTransformerScorer(model_loader=lambda *_: Model())
    with pytest.raises(JavaDocsRAGError, match="Reranker"):
        scorer([("q", "a"), ("q", "b")])


def test_model_load_failure_is_not_a_successful_score() -> None:
    def loader(*_):
        raise OSError("weights unavailable locally")

    with pytest.raises(JavaDocsRAGError, match="weights unavailable locally") as caught:
        SentenceTransformerScorer(model_loader=loader)([("q", "text")])
    assert isinstance(caught.value.__cause__, OSError)
