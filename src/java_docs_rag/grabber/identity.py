"""Developer B: stable document identifiers, URL normalization and content hashing.

Rules are recorded in docs/decisions/0001-indexing-identity-and-chunking.md.
"""

from __future__ import annotations

import hashlib
import posixpath
import re
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit

_JAVADOC_RE = re.compile(r"/javase/(?P<version>\d+)/docs/api/(?P<path>.+?)\.html?$")
_JAVA_VERSION_RE = re.compile(r"/javase/(?P<version>\d+)/")


def normalize_url(url: str, base_url: str | None = None) -> str:
    """Resolve, drop fragment and query, lowercase scheme/host, collapse dot segments."""
    absolute = urljoin(base_url, url) if base_url else url
    absolute, _fragment = urldefrag(absolute)
    parts = urlsplit(absolute)
    path = parts.path or "/"
    trailing = path.endswith("/")
    path = posixpath.normpath(path)
    if path == ".":
        path = "/"
    if trailing and not path.endswith("/"):
        path += "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def document_id_from_url(url: str) -> str:
    """Stable, human-readable ID that does not depend on content or run time.

    ``https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/List.html``
    becomes ``javase21:java.base/java/util/List``.
    """
    normalized = normalize_url(url)
    parts = urlsplit(normalized)
    match = _JAVADOC_RE.search(parts.path)
    if match:
        return f"javase{match.group('version')}:{match.group('path')}"
    path = re.sub(r"\.html?$", "", parts.path.strip("/"))
    return f"{parts.netloc}:{path}" if path else parts.netloc


def java_version_from_url(url: str) -> str | None:
    match = _JAVA_VERSION_RE.search(urlsplit(url).path)
    return match.group("version") if match else None


def content_hash(text: str | bytes) -> str:
    """SHA-256 hex digest; text is hashed as UTF-8."""
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()
