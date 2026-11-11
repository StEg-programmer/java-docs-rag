"""Developer B: normalize text while preserving Java identifiers and code.

Unicode NFC (not NFKC, which would rewrite symbols in code), special spaces to plain
spaces, zero-width characters removed, CRLF to LF, trailing spaces trimmed and runs
of blank lines collapsed. Case and identifiers are never changed; inside fenced code
blocks only line endings and special characters are touched.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

from java_docs_rag.common.models import Document
from java_docs_rag.grabber.identity import content_hash

_SPECIAL_SPACES = re.compile(r"[  -   　\t]")
_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿­]")
_FENCE = "```"


@dataclass
class DocumentNormalizer:
    collapse_inline_spaces: bool = True

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
        text = _ZERO_WIDTH.sub("", text)
        lines: list[str] = []
        in_code = False
        for line in text.split("\n"):
            if line.strip().startswith(_FENCE):
                in_code = not in_code
                lines.append(line.strip())
                continue
            if in_code:
                lines.append(line.replace(" ", " ").rstrip())
                continue
            line = _SPECIAL_SPACES.sub(" ", line)
            if self.collapse_inline_spaces:
                line = re.sub(r" {2,}", " ", line)
            lines.append(line.strip())
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    def normalize(self, document: Document) -> Document:
        text = self.normalize_text(document.content)
        title = " ".join(_SPECIAL_SPACES.sub(" ", document.title).split())
        return replace(document, title=title, content=text, content_hash=content_hash(text))
