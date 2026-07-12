from dataclasses import dataclass


@dataclass(slots=True)
class TokenUsage:
    """Aggregated token usage for an LLM execution."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
