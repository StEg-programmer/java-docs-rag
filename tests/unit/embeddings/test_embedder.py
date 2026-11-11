"""Developer B: SentenceTransformerEmbedder and factory with a fake model (no downloads)."""

import sys

import pytest

from java_docs_rag.common.config import EmbeddingConfig
from java_docs_rag.common.exceptions import ConfigurationError, EmbeddingError
from java_docs_rag.common.interfaces import Embedder
from java_docs_rag.embeddings.factory import KNOWN_MODELS, create_embedder, resolve_model_spec
from java_docs_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder


class FakeModel:
    def __init__(self, dimension: int = 4, returned_dimension: int | None = None) -> None:
        self.dimension = dimension
        self.returned = returned_dimension or dimension
        self.calls: list[tuple[list[str], dict]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return [[float(len(t))] + [0.0] * (self.returned - 1) for t in texts]


def make(model: FakeModel, **kwargs) -> tuple[SentenceTransformerEmbedder, list[str]]:
    loads: list[str] = []

    def loader(model_id: str, device: str, cache: str | None) -> FakeModel:
        loads.append(model_id)
        return model

    return SentenceTransformerEmbedder("fake/model", model_loader=loader, **kwargs), loads


def test_model_loads_lazily_once_and_satisfies_protocol() -> None:
    model = FakeModel()
    embedder, loads = make(model, dimension=4)
    assert isinstance(embedder, SentenceTransformerEmbedder)
    _: Embedder = embedder
    assert loads == [] and not embedder.is_loaded and embedder.dimension == 4
    embedder.embed_query("q")
    embedder.embed_documents(["a", "b"])
    assert loads == ["fake/model"]


def test_prefixes_batching_order_and_stats() -> None:
    model = FakeModel()
    embedder, _ = make(model, query_prefix="query: ", document_prefix="passage: ", batch_size=7)
    vectors = embedder.embed_documents(["a", "bbb"])
    assert [v[0] for v in vectors] == [len("passage: a"), len("passage: bbb")]
    assert embedder.embed_query("x")[0] == len("query: x")
    texts, kwargs = model.calls[0]
    assert texts == ["passage: a", "passage: bbb"]
    assert kwargs["batch_size"] == 7 and kwargs["normalize_embeddings"] is True
    assert embedder.stats.calls == 2 and embedder.stats.texts == 3
    assert embedder.embed_documents([]) == []


def test_dimension_from_model_and_mismatch_detection() -> None:
    embedder, _ = make(FakeModel(dimension=8))
    assert embedder.dimension == 8
    wrong, _ = make(FakeModel(dimension=4, returned_dimension=3), dimension=4)
    with pytest.raises(EmbeddingError, match="dimension 3"):
        wrong.embed_query("q")


def test_load_failure_is_wrapped() -> None:
    def broken(*_args):
        raise OSError("no network")

    embedder = SentenceTransformerEmbedder("x/y", 4, model_loader=broken)
    with pytest.raises(EmbeddingError, match="Cannot load"):
        embedder.embed_query("q")


def test_import_does_not_load_ml_libraries() -> None:
    assert "sentence_transformers" not in sys.modules
    assert "torch" not in sys.modules


def test_factory_presets_and_validation() -> None:
    e5 = create_embedder(EmbeddingConfig("e5-small"))
    assert e5.model_id == "intfloat/multilingual-e5-small" and e5.dimension == 384
    assert isinstance(e5, SentenceTransformerEmbedder)
    assert (e5.query_prefix, e5.document_prefix) == ("query: ", "passage: ")
    bge = create_embedder(EmbeddingConfig("BAAI/bge-small-en-v1.5", 384))
    assert isinstance(bge, SentenceTransformerEmbedder) and bge.document_prefix == ""
    assert resolve_model_spec("bge-small") is KNOWN_MODELS["BAAI/bge-small-en-v1.5"]
    custom = create_embedder(EmbeddingConfig("org/custom", 768))
    assert custom.dimension == 768
    for bad in (
        EmbeddingConfig(None),
        EmbeddingConfig("org/custom"),
        EmbeddingConfig("bge-small", 768),
    ):
        with pytest.raises(ConfigurationError):
            create_embedder(bad)
    assert not e5.is_loaded
