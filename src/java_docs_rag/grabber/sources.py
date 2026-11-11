"""Developer B: typed view of configs/sources.yaml (loaded only when called)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from java_docs_rag.common.exceptions import ConfigurationError


@dataclass(frozen=True)
class SourceConfig:
    source_id: str
    title: str
    base_url: str
    allowed_domains: list[str]
    allowed_path_prefixes: list[str] = field(default_factory=list)
    start_urls: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    max_pages: int | None = None
    language: str = "en"

    @property
    def seeds(self) -> list[str]:
        return list(self.start_urls) or [self.base_url]


def _as_list(value: object, key: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigurationError(f"'{key}' must be a list of strings")
    return list(value)


def parse_sources(data: object) -> list[SourceConfig]:
    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
        raise ConfigurationError("sources file must contain a 'sources' list")
    sources: list[SourceConfig] = []
    for raw in data["sources"]:
        if not isinstance(raw, dict):
            raise ConfigurationError("each source must be a mapping")
        missing = [key for key in ("source_id", "title", "base_url") if not raw.get(key)]
        if missing:
            raise ConfigurationError(f"source is missing required keys: {', '.join(missing)}")
        max_pages = raw.get("max_pages")
        if max_pages is not None and (not isinstance(max_pages, int) or max_pages <= 0):
            raise ConfigurationError("'max_pages' must be a positive integer or null")
        sources.append(
            SourceConfig(
                source_id=str(raw["source_id"]),
                title=str(raw["title"]),
                base_url=str(raw["base_url"]),
                allowed_domains=_as_list(raw.get("allowed_domains"), "allowed_domains"),
                allowed_path_prefixes=_as_list(
                    raw.get("allowed_path_prefixes"), "allowed_path_prefixes"
                ),
                start_urls=_as_list(raw.get("start_urls"), "start_urls"),
                exclude_patterns=_as_list(raw.get("exclude_patterns"), "exclude_patterns"),
                max_pages=max_pages,
                language=str(raw.get("language", "en")),
            )
        )
    ids = [source.source_id for source in sources]
    if len(ids) != len(set(ids)):
        raise ConfigurationError("source_id values must be unique")
    return sources


def load_sources(path: Path) -> list[SourceConfig]:
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"Cannot read sources file {path}: {exc}") from exc
    return parse_sources(data)


def get_source(sources: list[SourceConfig], source_id: str) -> SourceConfig:
    for source in sources:
        if source.source_id == source_id:
            return source
    raise ConfigurationError(f"Unknown source_id '{source_id}'")
