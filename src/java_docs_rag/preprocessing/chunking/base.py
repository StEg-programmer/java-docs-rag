"""Developer B: common base for configurable chunking strategies.

Conventions (docs/decisions/0001-indexing-identity-and-chunking.md):

* ``chunk_size`` and ``overlap`` are measured in characters;
* ``chunk_id = f"{document_id}#{strategy}-s{size}-o{overlap}#{position}"`` so IDs
  differ between chunking configurations but are stable between runs;
* ``section`` is the heading path (``"Interface List<E> > Method Details > size"``)
  in force at the chunk start, or ``None`` before the first heading;
* chunk metadata = document metadata + chunking parameters + character offsets.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from bisect import bisect_right
from dataclasses import dataclass
from typing import ClassVar

from java_docs_rag.common.config import ChunkingConfig
from java_docs_rag.common.exceptions import ConfigurationError
from java_docs_rag.common.models import Document, DocumentChunk

_HEADING = re.compile(r"^(#{1,6}) (.+)$")
_FENCE = "```"
SECTION_SEPARATOR = " > "


@dataclass(frozen=True)
class Span:
    start: int
    end: int


@dataclass(frozen=True)
class Heading:
    offset: int
    end: int
    level: int
    path: str


def find_headings(text: str) -> list[Heading]:
    """Markdown headings outside fenced code, with their hierarchical path."""
    headings: list[Heading] = []
    stack: list[tuple[int, str]] = []
    offset = 0
    in_code = False
    for line in text.split("\n"):
        if line.strip().startswith(_FENCE):
            in_code = not in_code
        elif not in_code:
            match = _HEADING.match(line)
            if match:
                level = len(match.group(1))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, match.group(2).strip()))
                path = SECTION_SEPARATOR.join(title for _, title in stack)
                headings.append(Heading(offset, offset + len(line), level, path))
        offset += len(line) + 1
    return headings


def find_blocks(text: str) -> list[Span]:
    """Paragraph-like blocks separated by blank lines; a fenced code block is one block."""
    blocks: list[Span] = []
    offset = 0
    start: int | None = None
    end = 0
    in_code = False
    for line in text.split("\n"):
        line_end = offset + len(line)
        if line.strip().startswith(_FENCE):
            in_code = not in_code
        if line.strip() or in_code:
            if start is None:
                start = offset
            end = line_end
        elif start is not None:
            blocks.append(Span(start, end))
            start = None
        offset = line_end + 1
    if start is not None:
        blocks.append(Span(start, end))
    return blocks


def window_spans(text: str, start: int, end: int, size: int, overlap: int) -> list[Span]:
    """Windows of at most ``size`` chars over ``text[start:end]``, preferring to break at
    whitespace; consecutive windows share about ``overlap`` chars."""
    spans: list[Span] = []
    position = start
    while position < end:
        while position < end and text[position].isspace():
            position += 1
        if position >= end:
            break
        limit = min(position + size, end)
        cut = limit
        if limit < end:
            space = max(text.rfind(" ", position, limit + 1), text.rfind("\n", position, limit + 1))
            if space > position + size // 2:
                cut = space
        piece_end = cut
        while piece_end > position and text[piece_end - 1].isspace():
            piece_end -= 1
        spans.append(Span(position, piece_end))
        if cut >= end:
            break
        next_position = max(cut - overlap, position + 1)
        if overlap and next_position > position + 1:
            # move to the start of a word so that windows do not begin mid-token
            boundary = text.find(" ", next_position - 1, cut)
            if boundary != -1 and boundary + 1 < cut:
                next_position = boundary + 1
        position = next_position
    return spans


@dataclass
class BaseChunker(ABC):
    config: ChunkingConfig
    strategy: ClassVar[str] = "base"
    uses_overlap: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.config.chunk_size <= 0:
            raise ConfigurationError("chunk_size must be positive")
        if self.uses_overlap and not 0 <= self.config.overlap < self.config.chunk_size:
            raise ConfigurationError("overlap must be in [0, chunk_size)")

    @property
    def chunk_size(self) -> int:
        return self.config.chunk_size

    @property
    def overlap(self) -> int:
        return self.config.overlap if self.uses_overlap else 0

    @property
    def config_tag(self) -> str:
        return f"{self.strategy}-s{self.chunk_size}-o{self.overlap}"

    @abstractmethod
    def spans(self, text: str) -> list[Span]:
        """Character spans of the chunks, in document order."""

    def split(self, document: Document) -> list[DocumentChunk]:
        """Split a document while retaining source identity and order."""
        text = document.content
        headings = find_headings(text)
        offsets = [h.offset for h in headings]
        by_offset = {h.offset: h for h in headings}
        chunks: list[DocumentChunk] = []
        for span in self.spans(text):
            piece = text[span.start : span.end].strip()
            if not piece:
                continue
            # A chunk that opens with headings belongs to the innermost of them.
            anchor = span.start
            while True:
                while anchor < span.end and text[anchor].isspace():
                    anchor += 1
                heading = by_offset.get(anchor)
                if heading is None or heading.end >= span.end:
                    break
                anchor = heading.end
            index = bisect_right(offsets, anchor) - 1
            section = headings[index].path if index >= 0 else None
            position = len(chunks)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.document_id}#{self.config_tag}#{position}",
                    document_id=document.document_id,
                    text=piece,
                    source_url=document.source_url,
                    title=document.title,
                    section=section,
                    position=position,
                    metadata={
                        **document.metadata,
                        "chunking_strategy": self.strategy,
                        "chunk_size": str(self.chunk_size),
                        "chunk_overlap": str(self.overlap),
                        "char_start": str(span.start),
                        "char_end": str(span.end),
                        "content_hash": document.content_hash,
                    },
                )
            )
        return chunks

    def pack_blocks(self, text: str, blocks: list[Span]) -> list[Span]:
        """Greedily merge consecutive blocks up to ``chunk_size``; split oversize ones."""
        spans: list[Span] = []
        current: Span | None = None
        for block in blocks:
            if block.end - block.start > self.chunk_size:
                if current:
                    spans.append(current)
                    current = None
                spans.extend(window_spans(text, block.start, block.end, self.chunk_size, 0))
            elif current is None:
                current = block
            elif block.end - current.start <= self.chunk_size:
                current = Span(current.start, block.end)
            else:
                spans.append(current)
                current = block
        if current:
            spans.append(current)
        return spans
