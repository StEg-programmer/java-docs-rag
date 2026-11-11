"""Developer B: download raw documents without parsing or indexing.

The HTTP client is injected; by default an ``httpx.Client`` is created lazily on the
first request, so importing or constructing the downloader performs no network I/O.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from java_docs_rag.common.exceptions import DocumentProcessingError

DEFAULT_USER_AGENT = "java-docs-rag/0.1 (university lab; polite crawler)"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class HttpResponse(Protocol):
    status_code: int
    content: bytes

    @property
    def headers(self) -> Mapping[str, str]: ...


class HttpClient(Protocol):
    def get(self, url: str) -> Any: ...


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    status_code: int
    content: bytes
    content_type: str

    @property
    def is_html(self) -> bool:
        return "html" in self.content_type.lower()

    def text(self, encoding: str = "utf-8") -> str:
        return self.content.decode(encoding, errors="replace")


def _default_client(user_agent: str, timeout: float) -> HttpClient:
    import httpx  # imported lazily: optional at import time, required for real downloads

    return httpx.Client(
        headers={"User-Agent": user_agent},
        timeout=timeout,
        follow_redirects=True,
    )


def _retryable_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [OSError, TimeoutError]
    try:
        import httpx

        errors.append(httpx.TransportError)
    except ImportError:  # pragma: no cover - httpx is a declared dependency
        pass
    return tuple(errors)


@dataclass
class DocumentDownloader:
    """Polite HTTP GET with retries, back-off and a minimum delay between requests."""

    client: HttpClient | None = None
    user_agent: str = DEFAULT_USER_AGENT
    timeout: float = 30.0
    max_retries: int = 3
    backoff_seconds: float = 1.0
    request_delay: float = 0.5
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    _last_request_at: float | None = field(default=None, init=False, repr=False)

    def _http(self) -> HttpClient:
        if self.client is None:
            self.client = _default_client(self.user_agent, self.timeout)
        return self.client

    def _respect_delay(self) -> None:
        if self._last_request_at is None or self.request_delay <= 0:
            return
        elapsed = self.clock() - self._last_request_at
        if elapsed < self.request_delay:
            self.sleep(self.request_delay - elapsed)

    def fetch(self, url: str) -> FetchResult:
        """GET ``url``; retries transport errors and 429/5xx, raises on final failure."""
        errors = _retryable_errors()
        last_problem = "unknown error"
        for attempt in range(self.max_retries + 1):
            if attempt:
                self.sleep(self.backoff_seconds * 2 ** (attempt - 1))
            self._respect_delay()
            try:
                response = self._http().get(url)
            except errors as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
                continue
            finally:
                self._last_request_at = self.clock()
            status = int(response.status_code)
            if status in RETRYABLE_STATUS:
                last_problem = f"HTTP {status}"
                continue
            if status >= 400:
                raise DocumentProcessingError(f"Cannot download {url}: HTTP {status}")
            headers = response.headers or {}
            return FetchResult(
                url=url,
                final_url=str(getattr(response, "url", url) or url),
                status_code=status,
                content=bytes(response.content),
                content_type=str(headers.get("content-type", "")),
            )
        raise DocumentProcessingError(
            f"Cannot download {url} after {self.max_retries + 1} attempts: {last_problem}"
        )

    def download(self, url: str, destination: Path) -> Path:
        """Fetch ``url`` and write the raw bytes to ``destination``."""
        result = self.fetch(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(result.content)
        return destination

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
