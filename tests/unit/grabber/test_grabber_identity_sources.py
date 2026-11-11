"""Developer B: URL normalization, document IDs, hashing and sources.yaml parsing."""

from pathlib import Path

import pytest

from java_docs_rag.common.exceptions import ConfigurationError
from java_docs_rag.grabber.identity import (
    content_hash,
    document_id_from_url,
    java_version_from_url,
    normalize_url,
)
from java_docs_rag.grabber.sources import get_source, load_sources, parse_sources

API = "https://docs.oracle.com/en/java/javase/21/docs/api/"


def test_normalize_url_resolves_relative_links_and_drops_fragment() -> None:
    page = API + "java.base/java/util/List.html"
    assert normalize_url("../lang/Iterable.html#method()", page) == (
        API + "java.base/java/lang/Iterable.html"
    )
    assert (
        normalize_url("HTTPS://Docs.Oracle.com/en/java/?q=1") == "https://docs.oracle.com/en/java/"
    )


def test_document_id_is_stable_and_readable() -> None:
    url = API + "java.base/java/util/List.html"
    assert document_id_from_url(url) == "javase21:java.base/java/util/List"
    assert document_id_from_url(url + "#size()") == document_id_from_url(url)
    assert document_id_from_url("https://example.org/a/b.html") == "example.org:a/b"


def test_java_version_and_hash() -> None:
    assert java_version_from_url(API) == "21"
    assert java_version_from_url("https://example.org/") is None
    assert content_hash("abc") == content_hash(b"abc")
    assert len(content_hash("abc")) == 64 and content_hash("abc") != content_hash("abd")


def test_repository_sources_file_is_valid() -> None:
    sources = load_sources(Path("configs/sources.yaml"))
    source = get_source(sources, "java-se-21-api")
    assert source.allowed_domains == ["docs.oracle.com"]
    assert all(url.startswith(source.base_url) for url in source.seeds)
    assert source.max_pages and source.max_pages > 0


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"sources": [{"title": "x", "base_url": "https://x"}]},
        {"sources": [{"source_id": "a", "title": "x", "base_url": "u", "max_pages": 0}]},
        {"sources": [{"source_id": "a", "title": "x", "base_url": "u", "allowed_domains": "d"}]},
        {"sources": [{"source_id": "a", "title": "x", "base_url": "u"}] * 2},
    ],
)
def test_invalid_sources_are_rejected(data: object) -> None:
    with pytest.raises(ConfigurationError):
        parse_sources(data)


def test_unknown_source_id() -> None:
    with pytest.raises(ConfigurationError):
        get_source([], "missing")
