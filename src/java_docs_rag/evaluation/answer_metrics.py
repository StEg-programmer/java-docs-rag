"""Developer A: answer quality and abstention evaluation without paid APIs."""

from java_docs_rag.common.models import EvaluationQuestion, RAGAnswer


def evaluate_answer(answer: RAGAnswer, question: EvaluationQuestion) -> dict[str, float]:
    raise NotImplementedError("Answer metrics are not implemented")
