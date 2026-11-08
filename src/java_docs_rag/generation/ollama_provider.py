"""Developer A: future local Ollama adapter; no requests on import or construction."""

from dataclasses import dataclass


@dataclass
class OllamaProvider:
    model_id: str
    base_url: str = "http://localhost:11434"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError("Ollama generation is not implemented")
