"""Standalone query preparation without adapters or external services."""

import pytest

from java_docs_rag.retrieval.query_processor import QueryProcessor


def test_plain_question_is_unchanged() -> None:
    query = "What is a Java List?"

    assert QueryProcessor().process(query) == query


@pytest.mark.parametrize("boundary", [" ", "\t", "\r\n", " \t\n", "\u2003"])
def test_boundary_whitespace_is_stripped(boundary: str) -> None:
    assert QueryProcessor().process(f"{boundary}What is a List?{boundary}") == "What is a List?"


def test_java_syntax_and_internal_whitespace_are_preserved() -> None:
    query = (
        "How does Map.Entry<String, Integer> work?\n"
        "    Map.Entry<String, Integer> entry = map.entrySet().iterator().next();\n"
        "\tSystem.out.println(entry.getKey());  // Keep  spacing"
    )

    assert QueryProcessor().process(f" \t\n{query}\n\t ") == query


@pytest.mark.parametrize("query", ["", " ", "\t", "\r\n", " \t\n ", "\u2003"])
def test_empty_or_whitespace_only_queries_are_rejected(query: str) -> None:
    with pytest.raises(ValueError, match="query must not be empty or whitespace-only"):
        QueryProcessor().process(query)


@pytest.mark.parametrize("query", [None, 42, True, b"List?", [], {}])
def test_non_string_queries_are_rejected(query: object) -> None:
    with pytest.raises(TypeError, match="query must be a string"):
        QueryProcessor().process(query)  # type: ignore[arg-type]
