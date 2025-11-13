from abc import ABC, abstractmethod
from typing import List

class BaseLLMClient(ABC):
    """Interface you can swap for a real LLM provider."""

    @abstractmethod
    def generate(self, question: str, context_chunks: List[str], max_tokens: int = 256) -> str:
        ...

class StubLLMClient(BaseLLMClient):
    """
    Minimal stub: concatenate small context snippets and echo an answer-like text.
    Replace this class with a real client (OpenAI, Anthropic, etc.).
    """
    def generate(self, question: str, context_chunks: List[str], max_tokens: int = 256) -> str:
        ctx = " ".join(c.strip().replace("\n", " ") for c in context_chunks)
        if len(ctx) > max_tokens:
            ctx = ctx[:max_tokens].rsplit(" ", 1)[0] + "..."
        return f"{ctx}"
