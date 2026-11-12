"""Developer A: deterministic evaluation with explicit applicable-question counts."""

from collections.abc import Sequence
from dataclasses import dataclass

from java_docs_rag.common.models import EvaluationQuestion
from java_docs_rag.evaluation.answer_metrics import evaluate_answer
from java_docs_rag.evaluation.retrieval_metrics import evaluate_retrieval
from java_docs_rag.pipeline.rag_pipeline import RAGPipeline


@dataclass
class EvaluationRunner:
    pipeline: RAGPipeline
    k: int = 10

    def run(self, questions: Sequence[EvaluationQuestion]) -> dict[str, float]:
        """Macro-average applicable metrics; report denominators, propagate all run errors.

        Retrieval includes only answerable questions with relevant IDs. Abstention
        includes all questions. Exact match includes answerable non-null references.
        An empty group has mean 0.0 and count 0, so absence is distinguishable.
        """
        # Validate even when the dataset is empty.
        evaluate_retrieval([], [], self.k)
        names = ("precision_at_k", "recall_at_k", "mrr_at_k", "abstention_accuracy", "exact_match")
        observations: dict[str, list[float]] = {name: [] for name in names}
        for question in questions:
            answer = self.pipeline.run(question.question)
            metrics = evaluate_answer(answer, question)
            if question.answerable and question.relevant_document_ids:
                metrics.update(
                    evaluate_retrieval(
                        answer.retrieved_chunks, question.relevant_document_ids, self.k
                    )
                )
            for name, value in metrics.items():
                observations[name].append(value)
        aggregates = {"question_count": float(len(questions))}
        for name, values in observations.items():
            aggregates[name] = sum(values) / len(values) if values else 0.0
            aggregates[f"{name}_count"] = float(len(values))
        return aggregates
