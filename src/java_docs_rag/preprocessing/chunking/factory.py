"""Developer B: build a Chunker from ChunkingConfig."""

from java_docs_rag.common.config import ChunkingConfig
from java_docs_rag.common.exceptions import ConfigurationError
from java_docs_rag.preprocessing.chunking.base import BaseChunker
from java_docs_rag.preprocessing.chunking.fixed import FixedChunker
from java_docs_rag.preprocessing.chunking.overlap import OverlapChunker
from java_docs_rag.preprocessing.chunking.paragraph import ParagraphChunker
from java_docs_rag.preprocessing.chunking.section import SectionChunker

CHUNKERS: dict[str, type[BaseChunker]] = {
    "fixed": FixedChunker,
    "overlap": OverlapChunker,
    "paragraph": ParagraphChunker,
    "section": SectionChunker,
}


def create_chunker(config: ChunkingConfig) -> BaseChunker:
    try:
        return CHUNKERS[config.strategy](config)
    except KeyError as exc:
        raise ConfigurationError(f"Unknown chunking strategy '{config.strategy}'") from exc
