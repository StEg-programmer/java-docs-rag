"""Developer A: grounded answer assembly and explicit abstention."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from java_docs_rag.common.exceptions import GenerationError
from java_docs_rag.common.interfaces import LLMProvider
from java_docs_rag.common.models import RAGAnswer, SearchResult
from java_docs_rag.generation.context_builder import ContextBuilder
from java_docs_rag.generation.prompt_builder import ABSTENTION_MARKER, PromptBuilder

INSUFFICIENT_ANSWER = "Insufficient context to answer the question."


@dataclass
class AnswerGenerator:
    provider: LLMProvider
    context_builder: ContextBuilder = field(default_factory=ContextBuilder)
    prompt_builder: PromptBuilder = field(default_factory=PromptBuilder)

    def generate(self, question: str, results: Sequence[SearchResult]) -> RAGAnswer:
        """Generate from selected evidence, or abstain without calling the provider.

        sources lists all chunks supplied to the model, even for a model-declared
        abstention; it does not certify citation use or factual support. A standalone
        call records selected results in retrieved_chunks; the pipeline replaces
        that field with its raw retrieval snapshot.
        """
        evidence = list(results)
        sources = [result.chunk for result in evidence]
        if not evidence:
            return RAGAnswer(question, INSUFFICIENT_ANSWER, [], [], self.provider.model_id, True)
        system, user = self.prompt_builder.build(question, self.context_builder.build(sources))
        content = self.provider.generate(system, user)
        if not isinstance(content, str) or not content.strip():
            raise GenerationError("Provider returned missing or blank answer content")
        content = content.strip()
        abstained = content == ABSTENTION_MARKER
        return RAGAnswer(
            question=question,
            answer=INSUFFICIENT_ANSWER if abstained else content,
            sources=sources,
            retrieved_chunks=evidence,
            model=self.provider.model_id,
            insufficient_context=abstained,
        )
