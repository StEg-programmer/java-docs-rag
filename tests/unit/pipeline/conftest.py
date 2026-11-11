"""Developer B: fakes shared by indexing pipeline and experiment tests."""

from collections.abc import Sequence

import pytest

from java_docs_rag.common.models import Document
from java_docs_rag.grabber.identity import content_hash


class FakeEmbedder:
    """Deterministic bag-of-letters embedding; no models involved."""

    model_id = "fake/letters"
    dimension = 26

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * 26
        for char in text.lower():
            if "a" <= char <= "z":
                vector[ord(char) - 97] += 1.0
        return vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def make_document(doc_id: str, text: str) -> Document:
    return Document(
        f"javase21:java.base/{doc_id}",
        f"https://x/{doc_id}.html",
        doc_id.title(),
        text,
        content_hash(text),
        metadata={"java_version": "21"},
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()
