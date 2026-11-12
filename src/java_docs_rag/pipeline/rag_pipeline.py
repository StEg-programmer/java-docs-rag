"""Developer A: synchronous retrieval, evidence selection and grounded generation."""

from dataclasses import dataclass, field, replace

from java_docs_rag.common.config import RetrievalConfig
from java_docs_rag.common.interfaces import LLMProvider, Reranker, Retriever
from java_docs_rag.common.models import RAGAnswer
from java_docs_rag.filtering.context_filter import ContextFilter
from java_docs_rag.filtering.deduplication import DeduplicationFilter
from java_docs_rag.filtering.similarity import SimilarityFilter
from java_docs_rag.generation.generator import AnswerGenerator
from java_docs_rag.observability.safe_tracing import SafeTelemetry
from java_docs_rag.retrieval.query_processor import QueryProcessor


@dataclass
class RAGPipeline:
    retriever: Retriever
    reranker: Reranker
    provider: LLMProvider
    config: RetrievalConfig = field(default_factory=RetrievalConfig)
    max_context_chars: int = 8000
    query_processor: QueryProcessor = field(default_factory=QueryProcessor)
    similarity_filter: SimilarityFilter = field(default_factory=SimilarityFilter)
    deduplication_filter: DeduplicationFilter = field(default_factory=DeduplicationFilter)
    context_filter: ContextFilter = field(default_factory=ContextFilter)
    answer_generator: AnswerGenerator | None = None
    tracer: SafeTelemetry = field(default_factory=SafeTelemetry)

    def run(self, question: str) -> RAGAnswer:
        """Prepare, retrieve, threshold, rerank, deduplicate, budget and generate.

        question is the prepared question. retrieved_chunks retains raw retrieval
        order/scores before all filters; sources is evidence supplied to the model.
        Adapter errors propagate, never becoming successful abstentions.
        """
        try:
            with self.tracer.span("rag.run") as root:
                query = self.query_processor.process(question)
                with self.tracer.span("rag.retrieve", {"top_k": self.config.top_k}) as span:
                    raw = list(self.retriever.retrieve(query, self.config.top_k))
                    span.update(output={"raw_count": len(raw)})
                with self.tracer.span("rag.similarity_filter") as span:
                    candidates = self.similarity_filter.filter(raw, self.config.min_similarity)
                    span.update(output={"eligible_count": len(candidates)})
                with self.tracer.span("rag.rerank", {"top_n": self.config.rerank_top_n}) as span:
                    ranked = self.reranker.rerank(query, candidates, self.config.rerank_top_n)
                    span.update(output={"ranked_count": len(ranked)})
                with self.tracer.span("rag.evidence_selection") as span:
                    unique = self.deduplication_filter.filter(ranked)
                    evidence = self.context_filter.filter(unique, self.max_context_chars)
                    span.update(
                        output={
                            "unique_count": len(unique),
                            "source_count": len(evidence),
                            "selected_chars": sum(len(item.chunk.text) for item in evidence),
                            "max_context_chars": self.max_context_chars,
                        }
                    )
                generator = self.answer_generator
                if generator is None:
                    generator = AnswerGenerator(self.provider)
                with self.tracer.span("rag.generate", {"provider_called": bool(evidence)}) as span:
                    answer = generator.generate(query, evidence)
                    outcome = "abstained" if answer.insufficient_context else "ok"
                    span.update(
                        output={
                            "insufficient_context": answer.insufficient_context,
                            "source_count": len(answer.sources),
                            "outcome": outcome,
                        }
                    )
                root.update(
                    output={
                        "raw_count": len(raw),
                        "source_count": len(answer.sources),
                        "insufficient_context": answer.insufficient_context,
                        "outcome": outcome,
                    }
                )
                return replace(answer, retrieved_chunks=raw)
        finally:
            self.tracer.flush()
