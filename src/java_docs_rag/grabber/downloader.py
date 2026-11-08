"""Developer B: download raw documents without parsing or indexing."""

from pathlib import Path


class DocumentDownloader:
    def download(self, url: str, destination: Path) -> Path:
        raise NotImplementedError("Document downloading is not implemented")
