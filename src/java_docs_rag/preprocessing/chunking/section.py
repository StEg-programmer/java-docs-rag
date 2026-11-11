"""Developer B: section chunking — one chunk per heading section.

A heading with no body of its own is merged into the following section, so
``## Method Details`` travels with its first method. Sections longer than
``chunk_size`` are split by paragraphs inside the section; sections are never merged.
"""

from java_docs_rag.preprocessing.chunking.base import (
    BaseChunker,
    Span,
    find_blocks,
    find_headings,
)


class SectionChunker(BaseChunker):
    strategy = "section"
    uses_overlap = False

    def spans(self, text: str) -> list[Span]:
        boundaries = [0, *(h.offset for h in find_headings(text) if h.offset > 0), len(text)]
        sections = [
            Span(start, end) for start, end in zip(boundaries, boundaries[1:], strict=False)
        ]
        headings = {h.offset: h for h in find_headings(text)}

        merged: list[Span] = []
        pending_start: int | None = None
        for section in sections:
            heading = headings.get(section.start)
            body = text[heading.end if heading else section.start : section.end]
            if heading and not body.strip():
                if pending_start is None:
                    pending_start = section.start
                continue
            start = pending_start if pending_start is not None else section.start
            pending_start = None
            if text[start : section.end].strip():
                merged.append(Span(start, section.end))
        if pending_start is not None:
            merged.append(Span(pending_start, len(text)))

        spans: list[Span] = []
        for section in merged:
            if section.end - section.start <= self.chunk_size:
                spans.append(section)
            else:
                blocks = [
                    Span(b.start + section.start, b.end + section.start)
                    for b in find_blocks(text[section.start : section.end])
                ]
                spans.extend(self.pack_blocks(text, blocks))
        return spans
