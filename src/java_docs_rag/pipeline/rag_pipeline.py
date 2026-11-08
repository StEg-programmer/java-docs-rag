"""Developer A: query orchestration boundary; no pipeline implementation."""

from dataclasses import dataclass

from java_docs_rag.common.interfaces import LLMProvider, Reranker, Retriever
from java_docs_rag.common.models import RAGAnswer


@dataclass
class RAGPipeline:
    retriever: Retriever
    reranker: Reranker
    provider: LLMProvider

    def run(self, question: str) -> RAGAnswer:
        raise NotImplementedError("RAG pipeline is not implemented")
