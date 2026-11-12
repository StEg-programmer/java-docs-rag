"""Cross-encoder behavior with a fake scorer and no model dependencies."""

from collections.abc import Sequence
from dataclasses import replace

import pytest

from java_docs_rag.common.interfaces import Reranker
from java_docs_rag.common.models import DocumentChunk, SearchResult
from java_docs_rag.reranking.cross_encoder import CrossEncoderReranker


class FakeScorer:
    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def __call__(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        self.calls.append(list(pairs))
        return self.scores


@pytest.fixture
def results() -> list[SearchResult]:
    return [
        SearchResult(
            chunk=DocumentChunk(
                chunk_id=f"java:{position}",
                document_id="java",
                text=text,
                source_url="https://example.test/java",
                title="Java APIs",
                section=None,
                position=position,
                metadata={"java_version": "21"},
            ),
            similarity_score=similarity,
        )
        for position, (text, similarity) in enumerate(
            [
                ("List is ordered.", 0.9),
                ("Map.Entry<String, Integer>", 0.7),
                ("Set is unique.", 0.5),
            ]
        )
    ]


@pytest.mark.parametrize("top_n", [1, 2, 3, 10])
def test_scores_every_candidate_then_limits_changed_ranking(
    results: list[SearchResult], top_n: int
) -> None:
    scorer = FakeScorer([0.1, 0.8, 0.5])
    reranker: Reranker = CrossEncoderReranker(model_id="fake-cross-encoder", scorer=scorer)
    query = " How does Map.Entry<String, Integer> work?\n"

    actual = reranker.rerank(query, results, top_n)

    assert scorer.calls == [
        [
            (query, "List is ordered."),
            (query, "Map.Entry<String, Integer>"),
            (query, "Set is unique."),
        ]
    ]
    expected = [(results[1], 0.8), (results[2], 0.5), (results[0], 0.1)][:top_n]
    assert len(actual) == len(expected)
    for updated, (original, score) in zip(actual, expected, strict=True):
        assert updated is not original
        assert updated.chunk is original.chunk
        assert updated.similarity_score == original.similarity_score
        assert updated.rerank_score == score


@pytest.mark.parametrize("top_n", [2, 10])
def test_ties_keep_original_order(results: list[SearchResult], top_n: int) -> None:
    reranker = CrossEncoderReranker("fake", FakeScorer([-1.0, 0.0, 0.0]))

    actual = reranker.rerank("Java?", tuple(results), top_n)

    assert [result.chunk for result in actual] == [
        results[index].chunk for index in [1, 2, 0][:top_n]
    ]


def test_original_sequence_and_results_are_untouched(results: list[SearchResult]) -> None:
    results[1] = replace(results[1], rerank_score=42.0)
    originals = list(results)
    snapshots = [replace(result) for result in results]
    reranker = CrossEncoderReranker("fake", FakeScorer([0.1, 0.8, 0.5]))

    updated = reranker.rerank("Java?", results, 3)

    assert results == snapshots
    assert all(result is original for result, original in zip(results, originals, strict=True))
    assert [result.rerank_score for result in results] == [None, 42.0, None]
    assert [result.rerank_score for result in updated] == [0.8, 0.5, 0.1]
    assert all(result.chunk.metadata == {"java_version": "21"} for result in results)
    assert reranker.model_id == "fake"


def test_empty_results_do_not_call_scorer() -> None:
    scorer = FakeScorer([])

    assert CrossEncoderReranker("fake", scorer).rerank("Java?", [], 1) == []
    assert scorer.calls == []


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize(
    ("top_n", "error_type"),
    [
        (True, TypeError),
        (False, TypeError),
        (None, TypeError),
        ("2", TypeError),
        (1.0, TypeError),
        (1.5, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_invalid_top_n_is_rejected_before_scoring(
    results: list[SearchResult], empty: bool, top_n: object, error_type: type[Exception]
) -> None:
    scorer = FakeScorer([0.1, 0.8, 0.5])

    with pytest.raises(error_type, match="top_n"):
        CrossEncoderReranker("fake", scorer).rerank(
            "Java?",
            [] if empty else results,
            top_n,  # type: ignore[arg-type]
        )

    assert scorer.calls == []


@pytest.mark.parametrize("scores", [[], [0.1, 0.2], [0.1, 0.2, 0.3, 0.4]])
def test_score_count_mismatch_is_rejected(results: list[SearchResult], scores: list[float]) -> None:
    scorer = FakeScorer(scores)

    with pytest.raises(ValueError, match=rf"{len(scores)} scores for 3 candidates"):
        CrossEncoderReranker("fake", scorer).rerank("Java?", results, 1)

    assert len(scorer.calls) == 1
    assert len(scorer.calls[0]) == 3
    assert all(result.rerank_score is None for result in results)
