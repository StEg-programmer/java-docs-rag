"""Developer A: future reproducible experiment runner."""

from collections.abc import Sequence
from dataclasses import dataclass

from java_docs_rag.common.models import EvaluationQuestion
from java_docs_rag.pipeline.rag_pipeline import RAGPipeline


@dataclass
class EvaluationRunner:
    pipeline: RAGPipeline

    def run(self, questions: Sequence[EvaluationQuestion]) -> dict[str, float]:
        raise NotImplementedError("Evaluation execution is not implemented")
