"""Developer B: remove navigation and other documentation boilerplate."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from java_docs_rag.common.models import Document
from java_docs_rag.grabber.identity import content_hash

DEFAULT_BOILERPLATE_PATTERNS: tuple[str, ...] = (
    r"^Skip navigation links$",
    r"^Report a bug or suggest an enhancement",
    r"^For further API reference and developer documentation",
    r"^Copyright ©",
    r"^Java is a trademark or registered trademark",
    r"^Scripting on this page tracks web page traffic",
    r"^Use is subject to license terms",
    r"^(Hide|Show) sidebar$",
    r"^(SEARCH|Search|Contents|Overview|Package|Class|Use|Tree|Deprecated|Index|Help)$",
    r"^(All Classes and Interfaces|All Packages|All Modules)$",
)
_FENCE = "```"


@dataclass
class DocumentCleaner:
    """Drops boilerplate lines, repeated lines and headings with no body.

    Lines inside fenced code blocks are never modified.
    """

    patterns: Sequence[str] = field(default_factory=lambda: DEFAULT_BOILERPLATE_PATTERNS)
    drop_empty_sections: bool = True

    def __post_init__(self) -> None:
        self._compiled = [re.compile(pattern) for pattern in self.patterns]

    def _is_boilerplate(self, line: str) -> bool:
        stripped = line.strip()
        return any(regex.search(stripped) for regex in self._compiled)

    def clean_text(self, text: str) -> str:
        kept: list[str] = []
        in_code = False
        previous = None
        for line in text.split("\n"):
            if line.strip().startswith(_FENCE):
                in_code = not in_code
                kept.append(line)
                previous = None
                continue
            if in_code:
                kept.append(line)
                continue
            if self._is_boilerplate(line):
                continue
            if line.strip() and line == previous:
                continue  # consecutive duplicate line (e.g. repeated nav label)
            kept.append(line)
            previous = line if line.strip() else previous
        cleaned = "\n".join(kept)
        if self.drop_empty_sections:
            cleaned = _drop_empty_sections(cleaned)
        return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    def clean(self, document: Document) -> Document:
        text = self.clean_text(document.content)
        return replace(document, content=text, content_hash=content_hash(text))


def _heading_level(line: str) -> int:
    match = re.match(r"^(#{1,6}) \S", line)
    return len(match.group(1)) if match else 0


def _drop_empty_sections(text: str) -> str:
    """Remove a heading whose section (until the next heading of the same or a higher
    level) contains no text — only blank lines or deeper headings that are also empty."""
    lines = text.split("\n")
    keep = [True] * len(lines)
    in_code = False
    code_lines = set()
    for i, line in enumerate(lines):
        if line.strip().startswith(_FENCE):
            in_code = not in_code
            code_lines.add(i)
        elif in_code:
            code_lines.add(i)
    for i in range(len(lines) - 1, -1, -1):
        level = 0 if i in code_lines else _heading_level(lines[i])
        if not level:
            continue
        has_body = False
        for j in range(i + 1, len(lines)):
            other = 0 if j in code_lines else _heading_level(lines[j])
            if other and other <= level:
                break
            if keep[j] and lines[j].strip() and not other:
                has_body = True
                break
            if keep[j] and other:
                has_body = True  # a kept (non-empty) subsection
                break
        if not has_body:
            keep[i] = False
    return "\n".join(line for line, k in zip(lines, keep, strict=True) if k)
