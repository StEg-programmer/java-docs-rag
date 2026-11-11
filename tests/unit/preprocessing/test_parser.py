"""Developer B: javadoc HTML → Document."""

from pathlib import Path

import pytest

from java_docs_rag.common.models import Document
from java_docs_rag.grabber.identity import content_hash
from java_docs_rag.preprocessing.parser import HTMLDocumentParser

FIXTURE = Path("tests/fixtures/javadoc/java.base/java/util/List.html")
URL = "https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/List.html"


@pytest.fixture(scope="module")
def document() -> Document:
    return HTMLDocumentParser().parse(FIXTURE.read_bytes(), URL + "#size()")


def test_identity_title_and_metadata(document: Document) -> None:
    assert document.document_id == "javase21:java.base/java/util/List"
    assert document.source_url == URL
    assert document.title == "Interface List<E>"
    assert document.updated_at == "2023-09-11"
    assert document.metadata == {
        "source": "oracle-javadoc",
        "java_version": "21",
        "module": "java.base",
        "package": "java.util",
        "kind": "interface",
        "type_name": "List",
        "page_type": "class-declaration-page",
    }
    assert document.content_hash == content_hash(document.content)


def test_navigation_scripts_and_footer_are_removed(document: Document) -> None:
    for noise in (
        "Skip navigation",
        "pathtoroot",
        "Copyright",
        "Report a bug",
        "Help",
        "All Methods",
    ):
        assert noise not in document.content


def test_structure_is_preserved(document: Document) -> None:
    content = document.content
    assert "# Interface List<E>" in content
    assert "## Method Summary" in content and "### size" in content
    assert "int | size() | Returns the number of elements in this list." in content
    assert "Returns: the number of elements in this list" in content
    # source line breaks inside a paragraph are joined
    assert "precise control over where in the list each element" in content


def test_code_blocks_keep_indentation(document: Document) -> None:
    expected = (
        '```\nList<String> names = List.of("a", "b");\n'
        "for (String name : names) {\n    System.out.println(name);\n}\n```"
    )
    assert expected in document.content


def test_summaries_can_be_excluded() -> None:
    doc = HTMLDocumentParser(include_summaries=False).parse(FIXTURE.read_bytes(), URL)
    assert "Method Summary" not in doc.content
    assert "### size" in doc.content


def test_minimal_page_falls_back_to_title_tag() -> None:
    html = (
        "<html><head><title>Overview (Java SE 21 &amp; JDK 21)</title></head>"
        "<body><p>Hello <b>world</b></p></body></html>"
    )
    doc = HTMLDocumentParser().parse(html, "https://example.org/x.html")
    assert doc.title == "Overview"
    assert doc.content == "Hello world"
    assert doc.updated_at is None
