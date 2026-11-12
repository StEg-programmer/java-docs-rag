"""Developer A: local Ollama adapter; no requests on import or construction."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from http.client import HTTPException
from math import isfinite
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from java_docs_rag.common.exceptions import GenerationError


@dataclass
class OllamaProvider:
    model_id: str
    base_url: str = "http://localhost:11434"
    timeout: float = 60.0
    transport: Callable[..., Any] = field(default=urlopen, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-blank string")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an HTTP(S) URL")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise TypeError("timeout must be a positive finite number")
        if not isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a positive finite number")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """POST a non-streaming chat request; wrap transport and response failures."""
        payload = {
            "model": self.model_id,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = Request(
            self.base_url.rstrip("/") + "/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.transport(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise GenerationError(f"Ollama returned HTTP {response.status}")
                data = json.loads(response.read().decode("utf-8"))
        except (URLError, OSError, HTTPException) as exc:
            raise GenerationError(f"Ollama transport failed: {exc}") from exc
        except (ValueError, UnicodeError) as exc:
            raise GenerationError("Ollama returned invalid UTF-8 JSON") from exc
        if not isinstance(data, dict) or data.get("error") or data.get("done") is not True:
            raise GenerationError("Ollama returned an error or incomplete chat response")
        message = data.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise GenerationError("Ollama response is missing an assistant message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise GenerationError("Ollama response has missing or blank content")
        return content.strip()
