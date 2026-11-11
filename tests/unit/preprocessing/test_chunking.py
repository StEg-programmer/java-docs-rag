"""Developer B: fixed / overlap / paragraph / section chunking."""

import pytest

from java_docs_rag.common.config import ChunkingConfig
from java_docs_rag.common.exceptions import ConfigurationError
from java_docs_rag.common.models import Document
from java_docs_rag.grabber.identity import content_hash
from java_docs_rag.preprocessing.chunking.base import find_blocks, find_headings
from java_docs_rag.preprocessing.chunking.factory import create_chunker

WORDS = " ".join(f"word{i:03d}" for i in range(200))  # 1599 chars, no newlines
STRUCTURED = "\n".join(
    [
        "Module java.base",
        "",
        "# Interface List<E>",
        "",
        "An ordered collection.",
        "",
        "```",
        "List<String> a = List.of();",
        "",
        "a.size();",
        "```",
        "",
        "## Method Details",
        "",
        "### size",
        "",
        "int size()",
        "",
        "Returns the size.",
        "",
        "### add",
        "",
        "boolean add(E e)",
        "",
        "Appends an element. " * 12,
    ]
)


def make(content: str) -> Document:
    return Document(
        "javase21:java.base/java/util/List",
        "https://x/List.html",
        "Interface List<E>",
        content,
        content_hash(content),
        metadata={"java_version": "21"},
    )


def chunker(strategy: str, size: int = 100, overlap: int = 20):
    return create_chunker(ChunkingConfig(strategy, size, overlap))


@pytest.mark.parametrize("strategy", ["fixed", "overlap", "paragraph", "section"])
def test_common_chunk_contract(strategy: str) -> None:
    document = make(STRUCTURED)
    chunks = chunker(strategy).split(document)
    assert chunks
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    overlap = 20 if strategy == "overlap" else 0
    assert chunks[0].chunk_id == f"{document.document_id}#{strategy}-s100-o{overlap}#0"
    for chunk in chunks:
        assert chunk.document_id == document.document_id
        assert chunk.source_url == document.source_url and chunk.title == document.title
        assert chunk.text and chunk.text == chunk.text.strip()
        assert chunk.metadata["java_version"] == "21"
        assert chunk.metadata["chunking_strategy"] == strategy
        assert chunk.metadata["content_hash"] == document.content_hash
        assert all(isinstance(v, str) for v in chunk.metadata.values())
    assert chunker(strategy).split(document) == chunks  # deterministic


@pytest.mark.parametrize("strategy", ["fixed", "overlap", "paragraph", "section"])
def test_chunks_respect_size_limit(strategy: str) -> None:
    for chunk in chunker(strategy, 120).split(make(STRUCTURED + "\n\n" + WORDS)):
        assert len(chunk.text) <= 120


def test_fixed_chunks_cover_text_without_overlap_or_split_words() -> None:
    chunks = chunker("fixed").split(make(WORDS))
    assert " ".join(c.text for c in chunks) == WORDS
    assert all(
        c.text.split()[0].startswith("word") and len(c.text.split()[-1]) == 7 for c in chunks
    )


def test_overlap_chunks_share_text() -> None:
    chunks = chunker("overlap", 100, 30).split(make(WORDS))
    assert len(chunks) > len(chunker("fixed", 100).split(make(WORDS)))
    for left, right in zip(chunks, chunks[1:], strict=False):
        assert right.text.split()[0] in left.text.split()


def test_overlap_validation() -> None:
    with pytest.raises(ConfigurationError):
        chunker("overlap", 100, 100)
    with pytest.raises(ConfigurationError):
        chunker("fixed", 0)
    with pytest.raises(ConfigurationError):
        create_chunker(ChunkingConfig("unknown", 100, 0))  # type: ignore[arg-type]
    assert chunker("fixed", 100, 500).overlap == 0  # overlap ignored outside "overlap"


def test_paragraph_keeps_code_block_whole() -> None:
    chunks = chunker("paragraph", 200).split(make(STRUCTURED))
    code = [c for c in chunks if "List.of()" in c.text]
    assert len(code) == 1 and "a.size();" in code[0].text


def test_section_chunks_follow_headings() -> None:
    chunks = chunker("section", 300).split(make(STRUCTURED))
    sections = [c.section for c in chunks]
    assert sections == [
        None,
        "Interface List<E>",
        "Interface List<E> > Method Details > size",
        "Interface List<E> > Method Details > add",
    ]
    # the empty "## Method Details" heading travels with its first subsection
    assert chunks[2].text.startswith("## Method Details\n\n### size")


def test_long_section_is_split_by_paragraphs() -> None:
    chunks = chunker("section", 120).split(make(STRUCTURED))
    add = [c for c in chunks if c.section and c.section.endswith("add")]
    assert len(add) >= 2


def test_helpers_ignore_headings_in_code() -> None:
    text = "# A\n\n```\n# not heading\n\nstill code\n```\n\n## B\ntext"
    assert [h.path for h in find_headings(text)] == ["A", "A > B"]
    blocks = [text[b.start : b.end] for b in find_blocks(text)]
    assert blocks[1] == "```\n# not heading\n\nstill code\n```"


def test_empty_document_has_no_chunks() -> None:
    assert chunker("paragraph").split(make("   \n\n ")) == []
