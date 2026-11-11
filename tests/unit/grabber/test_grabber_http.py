"""Developer B: downloader retries/politeness and crawler boundaries (no network)."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from java_docs_rag.common.exceptions import DocumentProcessingError
from java_docs_rag.grabber.crawler import DocumentationCrawler, extract_links
from java_docs_rag.grabber.downloader import DocumentDownloader, FetchResult

BASE = "https://docs.example.org/api/"


@dataclass
class FakeResponse:
    status_code: int
    content: bytes = b""
    headers: dict[str, str] = field(default_factory=lambda: {"content-type": "text/html"})
    url: str = ""


class ScriptedClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[str] = []

    def get(self, url: str) -> FakeResponse:
        self.requests.append(url)
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, FakeResponse)
        return item


def make_downloader(client: ScriptedClient, sleeps: list[float]) -> DocumentDownloader:
    return DocumentDownloader(
        client=client, request_delay=0, backoff_seconds=1, sleep=sleeps.append, max_retries=2
    )


def test_fetch_retries_transient_failures_with_backoff() -> None:
    sleeps: list[float] = []
    client = ScriptedClient([ConnectionError("reset"), FakeResponse(503), FakeResponse(200, b"ok")])
    result = make_downloader(client, sleeps).fetch(BASE)
    assert result.content == b"ok" and result.is_html
    assert sleeps == [1, 2]
    assert len(client.requests) == 3


def test_fetch_gives_up_after_retries_and_does_not_retry_404() -> None:
    sleeps: list[float] = []
    with pytest.raises(DocumentProcessingError, match="3 attempts"):
        make_downloader(ScriptedClient([FakeResponse(500)] * 3), sleeps).fetch(BASE)
    client = ScriptedClient([FakeResponse(404)])
    with pytest.raises(DocumentProcessingError, match="404"):
        make_downloader(client, sleeps).fetch(BASE)
    assert len(client.requests) == 1


def test_polite_delay_between_requests(tmp_path: Path) -> None:
    sleeps: list[float] = []
    now = [100.0]
    downloader = DocumentDownloader(
        client=ScriptedClient([FakeResponse(200, b"a"), FakeResponse(200, b"b")]),
        request_delay=0.5,
        sleep=sleeps.append,
        clock=lambda: now[0],
    )
    downloader.fetch(BASE)
    now[0] += 0.2
    path = downloader.download(BASE, tmp_path / "x" / "page.html")
    assert path.read_bytes() == b"b"
    assert sleeps == [pytest.approx(0.3)]


def page(*links: str) -> bytes:
    anchors = "".join(f'<a href="{link}">x</a>' for link in links)
    return f"<html><body>{anchors}</body></html>".encode()


class FakeSite:
    """Fetcher serving an in-memory site; unknown URLs are 404."""

    def __init__(self, pages: dict[str, bytes | str], robots: str | None = None) -> None:
        self.pages = pages
        self.robots = robots
        self.fetched: list[str] = []

    def fetch(self, url: str) -> FetchResult:
        self.fetched.append(url)
        if url.endswith("/robots.txt"):
            if self.robots is None:
                raise DocumentProcessingError("HTTP 404")
            return FetchResult(url, url, 200, self.robots.encode(), "text/plain")
        if url not in self.pages:
            raise DocumentProcessingError(f"Cannot download {url}: HTTP 404")
        body = self.pages[url]
        if isinstance(body, str):  # non-HTML resource
            return FetchResult(url, url, 200, body.encode(), "text/css")
        return FetchResult(url, url, 200, body, "text/html; charset=utf-8")


def test_extract_links_normalizes_and_deduplicates() -> None:
    html = page("b.html#x", "b.html", "mailto:a@b", "#top", "../other/c.html")
    assert extract_links(html, BASE + "pkg/a.html") == [BASE + "pkg/b.html", BASE + "other/c.html"]


def test_crawl_respects_domain_prefix_exclusions_and_budget() -> None:
    site = FakeSite(
        {
            BASE + "index.html": page(
                "a.html",
                "class-use/a.html",
                "https://evil.org/x.html",
                "../outside.html",
                "style.css",
                "missing.html",
            ),
            BASE + "a.html": page("index.html", "b.html"),
            BASE + "b.html": page(),
            BASE + "class-use/a.html": page(),
        }
    )
    crawler = DocumentationCrawler(site, exclude_patterns=[r"/class-use/"])
    urls = crawler.crawl(BASE + "index.html", ["docs.example.org"], ["/api/"])
    assert urls == [BASE + "index.html", BASE + "a.html", BASE + "b.html"]
    assert BASE + "missing.html" in crawler.failures
    assert all("evil" not in url and "class-use" not in url for url in site.fetched)

    limited = DocumentationCrawler(
        FakeSite(site.pages), max_pages=2, exclude_patterns=["class-use"]
    )
    assert len(limited.crawl(BASE + "index.html", ["docs.example.org"])) == 2


def test_crawl_honours_robots_txt() -> None:
    site = FakeSite(
        {
            BASE + "index.html": page("private/x.html", "a.html"),
            BASE + "a.html": page(),
            BASE + "private/x.html": page(),
        },
        robots="User-agent: *\nDisallow: /api/private/\n",
    )
    crawler = DocumentationCrawler(site)
    assert crawler.crawl(BASE + "index.html", ["docs.example.org"]) == [
        BASE + "index.html",
        BASE + "a.html",
    ]
    assert crawler.failures[BASE + "private/x.html"] == "disallowed by robots.txt"


def test_crawl_requires_a_start_url_inside_boundaries() -> None:
    with pytest.raises(DocumentProcessingError):
        DocumentationCrawler(FakeSite({})).crawl("https://other.org/", ["docs.example.org"])
