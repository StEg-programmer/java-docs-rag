"""Common exception hierarchy for future adapters and orchestration."""


class JavaDocsRAGError(Exception):
    """Base exception for application failures."""


class ConfigurationError(JavaDocsRAGError):
    """Invalid or missing configuration."""


class DocumentProcessingError(JavaDocsRAGError):
    """A source document could not be processed."""


class EmbeddingError(JavaDocsRAGError):
    """Embedding provider failure."""


class VectorStoreError(JavaDocsRAGError):
    """Vector storage failure."""


class GenerationError(JavaDocsRAGError):
    """Language model provider failure."""
