"""Developer B: boilerplate removal and text normalization."""

from java_docs_rag.common.models import Document
from java_docs_rag.grabber.identity import content_hash
from java_docs_rag.preprocessing.cleaner import DocumentCleaner
from java_docs_rag.preprocessing.normalizer import DocumentNormalizer


def doc(content: str, title: str = "T") -> Document:
    return Document("p:x", "https://x", title, content, content_hash(content))


def test_cleaner_drops_boilerplate_duplicates_and_empty_sections() -> None:
    text = "\n".join(
        [
            "Skip navigation links",
            "# List",
            "",
            "Body text.",
            "Body text.",
            "",
            "## Field Summary",
            "",
            "## Method Details",
            "",
            "### size",
            "",
            "Returns size.",
            "",
            "Copyright © 1993, 2023, Oracle",
            "Use is subject to license terms.",
        ]
    )
    cleaned = DocumentCleaner().clean(doc(text))
    assert cleaned.content == (
        "# List\n\nBody text.\n\n## Method Details\n\n### size\n\nReturns size."
    )
    assert cleaned.content_hash == content_hash(cleaned.content)


def test_cleaner_never_touches_code_blocks() -> None:
    text = "# Example\n\n```\n# not a heading\nSEARCH\nSEARCH\n```"
    assert DocumentCleaner().clean_text(text) == text


def test_cleaner_custom_patterns() -> None:
    assert DocumentCleaner(patterns=[r"^DRAFT$"]).clean_text("DRAFT\nkeep") == "keep"


def test_normalizer_whitespace_unicode_and_identifiers() -> None:
    text = "Map <K,V>  is​  a\tmap.  \r\n\r\n\r\n\r\nUse  HashMap.  "
    normalized = DocumentNormalizer().normalize(doc(text, title="Interface Map<K,V>"))
    assert normalized.content == "Map <K,V> is a map.\n\nUse HashMap."
    assert normalized.title == "Interface Map<K,V>"
    assert normalized.content_hash == content_hash(normalized.content)


def test_normalizer_keeps_code_indentation_and_nfc() -> None:
    text = "Café\n\n```\nif (x) {\n    y();   \n}\n```"
    assert DocumentNormalizer().normalize_text(text) == "Café\n\n```\nif (x) {\n    y();\n}\n```"
