"""Deterministic document metrics, answer checks and dataset/runner validation."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from java_docs_rag.common.models import EvaluationQuestion, RAGAnswer
from java_docs_rag.evaluation.answer_metrics import evaluate_answer
from java_docs_rag.evaluation.dataset import load_dataset
from java_docs_rag.evaluation.retrieval_metrics import evaluate_retrieval
from java_docs_rag.evaluation.runner import EvaluationRunner

FIXTURE = Path(__file__).parents[1] / "fixtures" / "evaluation" / "synthetic.json"


def question(**changes):
    return replace(
        EvaluationQuestion("q", "List?", "An ordered collection.", ["doc-1"], "api", True),
        **changes,
    )


def answer(candidates, **changes):
    return replace(
        RAGAnswer("List?", "An ordered collection.", [], list(candidates), "fake", False), **changes
    )


def test_dataset_fixture_is_utf8_and_matches_contract() -> None:
    questions = load_dataset(FIXTURE)
    assert len(questions) == 3
    assert questions[1].question == "Что такое Map.Entry<String, Integer>?"
    assert questions[1].expected_answer is None
    assert not questions[2].answerable


@pytest.mark.parametrize(
    "mutation",
    [
        {"question_id": ""},
        {"question": 42},
        {"category": " "},
        {"expected_answer": []},
        {"expected_answer": " "},
        {"answerable": 1},
        {"relevant_document_ids": "doc"},
        {"relevant_document_ids": [1]},
        {"relevant_document_ids": [""]},
        {"extra": "unknown"},
    ],
)
def test_invalid_dataset_fields(tmp_path, mutation) -> None:
    path = tmp_path / "questions.json"
    record = asdict(question()) | mutation
    path.write_text(json.dumps([record]), encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset record"):
        load_dataset(path)


@pytest.mark.parametrize("content", ["{}", "null", "[1]", "[{}]", "bad JSON"])
def test_invalid_dataset_structure(tmp_path, content) -> None:
    path = tmp_path / "questions.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


def test_duplicate_question_ids_are_rejected(tmp_path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([asdict(question())] * 2), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate question_id"):
        load_dataset(path)


def test_dataset_empty_and_missing_file(tmp_path) -> None:
    path = tmp_path / "questions.json"
    path.write_text("[]", encoding="utf-8")
    assert load_dataset(path) == []
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.json")


def test_retrieval_deduplicates_documents_before_cutoff(candidates) -> None:
    duplicate_doc = replace(candidates[0], chunk=replace(candidates[0].chunk, chunk_id="other"))
    metrics = evaluate_retrieval(
        [candidates[0], duplicate_doc, *candidates[1:]], ["doc-1", "doc-1", "doc-2"], 2
    )
    assert metrics == {"precision_at_k": 0.5, "recall_at_k": 0.5, "mrr_at_k": 0.5}


def test_precision_denominator_is_k_even_when_results_are_short(candidates) -> None:
    assert evaluate_retrieval(candidates[:1], ["doc-0"], 4) == {
        "precision_at_k": 0.25,
        "recall_at_k": 1.0,
        "mrr_at_k": 1.0,
    }


def test_empty_retrieval_or_relevance(candidates) -> None:
    zeros = {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr_at_k": 0.0}
    assert evaluate_retrieval([], ["doc-0"], 1) == zeros
    assert evaluate_retrieval(candidates, [], 1) == zeros
    assert evaluate_retrieval(candidates, ["missing"], 2) == zeros


@pytest.mark.parametrize(
    ("k", "error"), [(True, TypeError), (1.0, TypeError), (0, ValueError), (-1, ValueError)]
)
def test_invalid_k(k, error) -> None:
    with pytest.raises(error, match="k"):
        evaluate_retrieval([], [], k)


@pytest.mark.parametrize(
    ("text", "match"), [(" AN\nordered\tCOLLECTION. ", 1.0), ("An ordered collection", 0.0)]
)
def test_answer_exact_match_normalizes_case_and_whitespace(candidates, text, match) -> None:
    assert evaluate_answer(answer(candidates, answer=text), question()) == {
        "abstention_accuracy": 1.0,
        "exact_match": match,
    }


def test_missing_reference_is_not_an_exact_match_observation(candidates) -> None:
    assert evaluate_answer(answer(candidates), question(expected_answer=None)) == {
        "abstention_accuracy": 1.0,
    }


def test_unanswerable_and_wrong_abstention(candidates) -> None:
    abstained = answer(candidates, insufficient_context=True)
    assert evaluate_answer(abstained, question(answerable=False)) == {"abstention_accuracy": 1.0}
    assert evaluate_answer(answer(candidates), question(answerable=False)) == {
        "abstention_accuracy": 0.0,
    }
    assert evaluate_answer(abstained, question()) == {
        "abstention_accuracy": 0.0,
        "exact_match": 0.0,
    }


def test_runner_uses_raw_retrieval_and_applicable_denominators(candidates) -> None:
    questions = [
        question(),
        question(question_id="q2", expected_answer=None),
        question(question_id="q3", answerable=False, relevant_document_ids=[]),
    ]

    class Pipeline:
        def __init__(self):
            self.calls = []

        def run(self, text):
            self.calls.append(text)
            # sources deliberately unrelated to the relevant raw retrieval document.
            return answer(
                candidates, sources=[candidates[0].chunk], insufficient_context=len(self.calls) == 3
            )

    pipeline = Pipeline()
    metrics = EvaluationRunner(pipeline, k=2).run(questions)
    assert pipeline.calls == [q.question for q in questions]
    assert metrics["precision_at_k"] == 0.5
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr_at_k"] == 0.5
    assert metrics["precision_at_k_count"] == 2
    assert metrics["exact_match"] == 1.0
    assert metrics["exact_match_count"] == 1
    assert metrics["abstention_accuracy"] == 1.0
    assert metrics["abstention_accuracy_count"] == metrics["question_count"] == 3


def test_runner_empty_dataset_and_inapplicable_groups(candidates) -> None:
    class Pipeline:
        def run(self, text):
            return answer(candidates, insufficient_context=True)

    runner = EvaluationRunner(Pipeline())
    assert all(value == 0 for value in runner.run([]).values())
    metrics = runner.run([question(answerable=False, relevant_document_ids=[])])
    assert metrics["exact_match_count"] == metrics["recall_at_k_count"] == 0
    assert metrics["abstention_accuracy"] == 1.0


def test_runner_surfaces_execution_errors() -> None:
    class Pipeline:
        def run(self, text):
            raise RuntimeError("pipeline failed")

    with pytest.raises(RuntimeError, match="pipeline failed"):
        EvaluationRunner(Pipeline()).run([question()])
