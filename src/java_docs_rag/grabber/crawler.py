"""Developer B: discover documentation URLs within configured source boundaries.

Breadth-first crawl limited by domain, path prefixes, exclusion patterns, robots.txt
and a page budget. Pages are yielded as they are fetched so the caller can store
them without downloading each page twice.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup

from java_docs_rag.common.exceptions import DocumentProcessingError
from java_docs_rag.grabber.downloader import DEFAULT_USER_AGENT, FetchResult
from java_docs_rag.grabber.identity import normalize_url

logger = logging.getLogger(__name__)

_PAGE_SUFFIXES = (".html", ".htm", "/")


class Fetcher(Protocol):
    def fetch(self, url: str) -> FetchResult: ...


def extract_links(html: str | bytes, page_url: str) -> list[str]:
    """Absolute, normalized ``<a href>`` targets in document order, without duplicates."""
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not href or href.startswith(("mailto:", "javascript:", "#")):
            continue
        url = normalize_url(href, page_url)
        if url not in seen:
            seen.add(url)
            links.append(url)
    return links


@dataclass
class DocumentationCrawler:
    fetcher: Fetcher | None = None
    max_pages: int | None = None
    exclude_patterns: Sequence[str] = ()
    respect_robots: bool = True
    user_agent: str = DEFAULT_USER_AGENT
    failures: dict[str, str] = field(default_factory=dict, init=False)
    _robots: dict[str, RobotFileParser | None] = field(default_factory=dict, init=False)

    def _fetcher(self) -> Fetcher:
        if self.fetcher is None:
            from java_docs_rag.grabber.downloader import DocumentDownloader

            self.fetcher = DocumentDownloader(user_agent=self.user_agent)
        return self.fetcher

    def is_allowed(
        self,
        url: str,
        allowed_domains: Sequence[str],
        allowed_path_prefixes: Sequence[str] = (),
    ) -> bool:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"}:
            return False
        host = parts.hostname or ""
        if not any(host == d or host.endswith("." + d) for d in allowed_domains):
            return False
        if allowed_path_prefixes and not any(
            parts.path.startswith(p) for p in allowed_path_prefixes
        ):
            return False
        if not parts.path.endswith(_PAGE_SUFFIXES):
            return False
        return not any(re.search(pattern, url) for pattern in self.exclude_patterns)

    def _robots_allows(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            parser: RobotFileParser | None = RobotFileParser()
            try:
                result = self._fetcher().fetch(origin + "/robots.txt")
                parser.parse(result.text().splitlines())
            except DocumentProcessingError:
                parser = None  # robots.txt unavailable (e.g. 404): no restrictions known
            self._robots[origin] = parser
        parser = self._robots[origin]
        return parser is None or parser.can_fetch(self.user_agent, url)

    def iter_pages(
        self,
        base_url: str,
        allowed_domains: Sequence[str],
        allowed_path_prefixes: Sequence[str] = (),
        start_urls: Sequence[str] = (),
    ) -> Iterator[FetchResult]:
        """Yield fetched HTML pages in BFS order; failed pages are recorded, not raised."""
        self.failures = {}
        queue: deque[str] = deque()
        seen: set[str] = set()
        for seed in start_urls or [base_url]:
            url = normalize_url(seed)
            if url not in seen and self.is_allowed(url, allowed_domains, allowed_path_prefixes):
                seen.add(url)
                queue.append(url)
        if not queue:
            raise DocumentProcessingError("No start URL is inside the allowed boundaries")

        fetched = 0
        while queue and (self.max_pages is None or fetched < self.max_pages):
            url = queue.popleft()
            if not self._robots_allows(url):
                self.failures[url] = "disallowed by robots.txt"
                continue
            try:
                page = self._fetcher().fetch(url)
            except DocumentProcessingError as exc:
                self.failures[url] = str(exc)
                logger.warning("crawl failed for %s: %s", url, exc)
                continue
            if page.content_type and not page.is_html:
                continue
            final_url = normalize_url(page.final_url)
            if final_url != url and not self.is_allowed(
                final_url, allowed_domains, allowed_path_prefixes
            ):
                self.failures[url] = f"redirected outside boundaries to {final_url}"
                continue
            fetched += 1
            yield page
            for link in extract_links(page.content, final_url):
                if link not in seen and self.is_allowed(
                    link, allowed_domains, allowed_path_prefixes
                ):
                    seen.add(link)
                    queue.append(link)

    def crawl(
        self,
        base_url: str,
        allowed_domains: Sequence[str],
        allowed_path_prefixes: Sequence[str] = (),
        start_urls: Sequence[str] = (),
    ) -> list[str]:
        """Return discovered page URLs (in fetch order)."""
        return [
            normalize_url(page.url)
            for page in self.iter_pages(
                base_url, allowed_domains, allowed_path_prefixes, start_urls
            )
        ]
