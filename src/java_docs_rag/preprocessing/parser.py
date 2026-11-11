"""Developer B: parse downloaded HTML into a common Document.

The text keeps document structure in a light Markdown form that later stages rely on:

* headings become ``#``..``######`` lines (used by section chunking and ``section``);
* ``<pre>`` blocks become fenced code blocks and keep their whitespace;
* list items become ``- `` lines, summary tables become ``a | b | c`` rows.

Navigation, header/footer, scripts and buttons are dropped here because they are
recognisable only in HTML; textual boilerplate is removed later by DocumentCleaner.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag

from java_docs_rag.common.models import Document
from java_docs_rag.grabber.identity import (
    content_hash,
    document_id_from_url,
    java_version_from_url,
    normalize_url,
)

_DROP_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "button",
    "img",
    "svg",
    "form",
    "input",
    "template",
)
_DROP_SELECTORS = (
    "[role=navigation]",
    "[role=banner]",
    "[role=contentinfo]",
    ".sub-nav",
    ".top-nav",
    ".skip-nav",
    ".nav-list",
    "#navbar-top",
    ".legal-copy",
    ".toc",
    "#toc",
    ".flex-header",
)
_BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "aside",
    "dl",
    "ul",
    "ol",
    "table",
    "tr",
    "blockquote",
    "hr",
    "main",
    "body",
    "details",
    "summary",
    "caption",
}
_TITLE_SUFFIX = re.compile(r"\s*\((Java SE|Java Platform|JDK)[^)]*\)\s*$")
_DECLARATION = re.compile(
    r"(module|package|class|interface|enum|record|annotation interface|"
    r"exception|error|annotation type):\s*([\w.$<>]+)"
)
_PLACEHOLDER = "\x00CODE{}\x00"


def _text(node: Tag) -> str:
    return " ".join(node.get_text("").split())


@dataclass
class HTMLDocumentParser:
    include_summaries: bool = True
    source_name: str = "oracle-javadoc"

    def parse(self, html: str | bytes, source_url: str) -> Document:
        url = normalize_url(source_url)
        soup = BeautifulSoup(html, "html.parser")
        title = self._title(soup)
        updated_at = self._meta(soup, "dc.created")
        metadata = self._metadata(soup, url)

        for tag in soup.find_all(_DROP_TAGS):
            tag.decompose()
        for selector in _DROP_SELECTORS:
            for tag in soup.select(selector):
                tag.decompose()
        if not self.include_summaries:
            for tag in soup.select("section.summary"):
                tag.decompose()

        root = soup.find("main") or soup.body or soup
        content = self._render(root)
        return Document(
            document_id=document_id_from_url(url),
            source_url=url,
            title=title,
            content=content,
            content_hash=content_hash(content),
            updated_at=updated_at,
            metadata=metadata,
        )

    # -- metadata -----------------------------------------------------------------

    @staticmethod
    def _meta(soup: BeautifulSoup, name: str) -> str | None:
        tag = soup.find("meta", attrs={"name": name})
        value = tag.get("content") if isinstance(tag, Tag) else None
        return str(value).strip() if value else None

    def _title(self, soup: BeautifulSoup) -> str:
        heading = soup.find("h1")
        if isinstance(heading, Tag) and _text(heading):
            return _text(heading)
        if soup.title and soup.title.string:
            return _TITLE_SUFFIX.sub("", soup.title.string.strip())
        return ""

    def _metadata(self, soup: BeautifulSoup, url: str) -> dict[str, str]:
        metadata = {"source": self.source_name}
        version = java_version_from_url(url)
        if version:
            metadata["java_version"] = version
        description = self._meta(soup, "description") or ""
        for kind, name in _DECLARATION.findall(description):
            if kind in {"module", "package"}:
                metadata[kind] = name
            else:
                metadata["kind"] = kind
                metadata["type_name"] = name
        body = soup.body
        classes = body.get("class") if isinstance(body, Tag) else None
        if classes:
            metadata["page_type"] = str(classes[0])
        return metadata

    # -- rendering ----------------------------------------------------------------

    def _render(self, root: Tag) -> str:
        code_blocks: list[str] = []

        # HTML whitespace (including source line breaks) is insignificant outside <pre>.
        for string in list(root.find_all(string=True)):
            if "\n" in string and string.find_parent("pre") is None:
                string.replace_with(NavigableString(string.replace("\n", " ")))

        for pre in root.find_all("pre"):
            code = textwrap.dedent(pre.get_text().rstrip()).strip("\n")
            pre.replace_with(
                NavigableString("\n\n" + _PLACEHOLDER.format(len(code_blocks)) + "\n\n")
            )
            code_blocks.append(code)

        for table in root.select("div.summary-table"):
            table.replace_with(NavigableString("\n\n" + self._summary_table(table) + "\n\n"))

        for heading in root.find_all(re.compile(r"^h[1-6]$")):
            level = int(heading.name[1])
            text = _text(heading)
            heading.replace_with(
                NavigableString(f"\n\n{'#' * level} {text}\n\n" if text else "\n\n")
            )

        for tag in root.find_all(["dt", "dd", "li", "br", "td", "th"]):
            if tag.name == "dt":
                tag.insert_before(NavigableString("\n"))
                tag.insert_after(NavigableString(" "))
            elif tag.name == "dd":
                tag.insert_after(NavigableString("\n"))
            elif tag.name == "li":
                tag.insert_before(NavigableString("\n- "))
                tag.insert_after(NavigableString("\n"))
            elif tag.name == "br":
                tag.replace_with(NavigableString("\n"))
            else:  # table cells
                tag.insert_after(NavigableString(" | "))

        for tag in root.find_all(_BLOCK_TAGS):
            tag.insert_before(NavigableString("\n\n"))
            tag.insert_after(NavigableString("\n\n"))

        raw = root.get_text("")
        lines = [" ".join(line.split()) for line in raw.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"^- ?$", "", text, flags=re.MULTILINE)  # list markers left empty
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        for index, code in enumerate(code_blocks):
            text = text.replace(_PLACEHOLDER.format(index), f"```\n{code}\n```")
        return text

    @staticmethod
    def _summary_table(table: Tag) -> str:
        classes = table.get("class") or []
        columns = 3
        for name in classes:
            match = re.match(r"(two|three|four)-column-summary", str(name))
            if match:
                columns = {"two": 2, "three": 3, "four": 4}[match.group(1)]
        cells = [
            _text(cell)
            for cell in table.find_all("div", recursive=False)
            if any(str(c).startswith(("col-", "table-header")) for c in (cell.get("class") or []))
        ]
        rows = [cells[i : i + columns] for i in range(0, len(cells), columns)]
        return "\n".join(" | ".join(cell for cell in row) for row in rows if any(row))
