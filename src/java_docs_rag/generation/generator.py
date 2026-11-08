"""Developer A: future grounded answer assembly."""

from collections.abc import Sequence
from dataclasses import dataclass

from java_docs_rag.common.interfaces import LLMProvider
from java_docs_rag.common.models import RAGAnswer, SearchResult


@dataclass
class AnswerGenerator:
    provider: LLMProvider

    def generate(self, question: str, results: Sequence[SearchResult]) -> RAGAnswer:
        raise NotImplementedError("Answer generation is not implemented")
