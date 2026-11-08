"""Developer B: discover documentation URLs within configured source boundaries."""

from collections.abc import Sequence


class DocumentationCrawler:
    def crawl(self, base_url: str, allowed_domains: Sequence[str]) -> list[str]:
        raise NotImplementedError("Documentation crawling is not implemented")
