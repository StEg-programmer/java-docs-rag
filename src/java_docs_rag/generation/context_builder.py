"""Developer A: format selected chunks with traceable source identifiers."""

import json
from collections.abc import Sequence

from java_docs_rag.common.models import DocumentChunk


class ContextBuilder:
    def build(self, chunks: Sequence[DocumentChunk]) -> str:
        """Serialize evidence as JSON with stable, one-based citation labels [S1], ... ."""
        return json.dumps(
            [
                {
                    "citation": f"[S{index}]",
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "source_url": chunk.source_url,
                    "title": chunk.title,
                    "section": chunk.section,
                    "text": chunk.text,
                }
                for index, chunk in enumerate(chunks, start=1)
            ],
            ensure_ascii=False,
        )
