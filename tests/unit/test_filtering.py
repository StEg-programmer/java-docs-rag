"""Filtering uses retrieval scores, preserves rank, and budgets whole chunk text."""

from dataclasses import replace

import pytest

from java_docs_rag.common.models import SearchResult
from java_docs_rag.filtering.context_filter import ContextFilter
from java_docs_rag.filtering.deduplication import DeduplicationFilter
from java_docs_rag.filtering.similarity import SimilarityFilter


@pytest.mark.parametrize(("threshold", "indices"), [(None, [0, 1, 2]), (0.5, [0, 1]), (1, [])])
def test_similarity_cutoff_is_optional_and_inclusive(candidates, threshold, indices) -> None:
    original = list(candidates)
    actual = SimilarityFilter().filter(tuple(candidates), threshold)
    assert actual == [candidates[index] for index in indices]
    assert candidates == original
    assert all(result is candidates[index] for result, index in zip(actual, indices, strict=True))


@pytest.mark.parametrize("threshold", [float("nan"), float("inf")])
def test_nonfinite_threshold_is_rejected(threshold) -> None:
    with pytest.raises(ValueError, match="finite"):
        SimilarityFilter().filter([], threshold)


@pytest.mark.parametrize("threshold", [True, "0.5"])
def test_invalid_threshold_type_is_rejected(threshold) -> None:
    with pytest.raises(TypeError, match="threshold"):
        SimilarityFilter().filter([], threshold)


def test_deduplication_keeps_first_ranked_chunk(candidates: list[SearchResult]) -> None:
    duplicate = replace(candidates[0], rerank_score=999.0)
    ranked = [candidates[1], candidates[0], duplicate, candidates[2]]
    actual = DeduplicationFilter().filter(ranked)
    assert actual == [candidates[1], candidates[0], candidates[2]]
    assert actual[1] is candidates[0]
    assert ranked[2] is duplicate


@pytest.mark.parametrize(
    ("budget", "indices"), [(0, []), (3, [2]), (4, [0]), (7, [0, 2]), (16, [0, 1]), (19, [0, 1, 2])]
)
def test_context_exact_fits_and_skips_oversized_chunks(candidates, budget, indices) -> None:
    assert ContextFilter().filter(candidates, budget) == [candidates[index] for index in indices]


def test_budget_counts_unicode_and_whitespace_only_in_text(candidates) -> None:
    result = replace(candidates[0], chunk=replace(candidates[0].chunk, text="Я\n\t"))
    assert ContextFilter().filter([result], 3) == [result]
    assert ContextFilter().filter([result], 2) == []


@pytest.mark.parametrize(
    ("budget", "error"), [(-1, ValueError), (True, TypeError), (1.5, TypeError), (None, TypeError)]
)
def test_invalid_budget_is_rejected_even_with_empty_input(budget, error) -> None:
    with pytest.raises(error, match="max_chars"):
        ContextFilter().filter([], budget)


def test_empty_inputs() -> None:
    assert SimilarityFilter().filter([], 0.5) == []
    assert DeduplicationFilter().filter([]) == []
    assert ContextFilter().filter([], 0) == []
