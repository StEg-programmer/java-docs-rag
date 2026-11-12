"""Developer A: answer quality and abstention evaluation without paid APIs."""

from java_docs_rag.common.models import EvaluationQuestion, RAGAnswer


def evaluate_answer(answer: RAGAnswer, question: EvaluationQuestion) -> dict[str, float]:
    """Check abstention for every question and exact match only for answerable references.

    Exact match casefolds and collapses Unicode whitespace; punctuation remains.
    An abstention on an answerable reference is always an exact-match failure.
    Missing metrics are inapplicable, not zero-valued observations.
    """
    metrics = {
        "abstention_accuracy": float(answer.insufficient_context == (not question.answerable))
    }
    if question.answerable and question.expected_answer is not None:

        def normalize(text: str) -> str:
            return " ".join(text.casefold().split())

        metrics["exact_match"] = float(
            not answer.insufficient_context
            and normalize(answer.answer) == normalize(question.expected_answer)
        )
    return metrics
