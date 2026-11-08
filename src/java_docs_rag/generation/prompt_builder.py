"""Developer A: prompts requiring grounded answers and an abstention path."""


class PromptBuilder:
    def build(self, question: str, context: str) -> tuple[str, str]:
        """Return system and user prompts."""
        raise NotImplementedError("Prompt building is not implemented")
