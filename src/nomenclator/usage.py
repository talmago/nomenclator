from dataclasses import dataclass

import dspy

from nomenclator.exceptions import HSInitializationError


@dataclass(slots=True)
class TokenUsage:
    """Aggregated token usage for an LLM execution."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0


def calc_usage(
    history: list[dict],
) -> TokenUsage:
    """Aggregate token usage across DSPy language model calls.

    This method summarizes prompt tokens, completion tokens, total tokens,
    and estimated cost from the recorded DSPy language model execution
    history. Missing usage information is treated as zero, allowing the
    method to work across different providers and DSPy/LiteLLM versions.

    Args:
        history: Sequence of language model call records, typically obtained
            from ``dspy.LM.history``.

    Returns:
        Aggregated token usage statistics for the complete classification
        pipeline.
    """
    usage = TokenUsage()

    for call in history:
        usage.cost += float(call.get("cost", 0.0))

        call_usage = call.get("usage") or {}

        usage.prompt_tokens += int(call_usage.get("prompt_tokens", 0))
        usage.completion_tokens += int(call_usage.get("completion_tokens", 0))
        usage.total_tokens += int(call_usage.get("total_tokens", 0))

    if usage.total_tokens == 0:
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

    return usage


def ensure_dspy_lm() -> None:
    """Raise if DSPy has no usable language model configured.

    The classification pipeline's LLM stages (Product Analyst, Research
    Analyst, Classification Analyst) read ``dspy.settings.lm`` at call time.
    Failing early with a clear initialization error avoids wrapping a missing
    LM configuration as a product-analysis failure.
    """

    lm = dspy.settings.lm

    if lm is None:
        raise HSInitializationError(
            "DSPy language model is not configured. "
            "Call dspy.configure(lm=dspy.LM(...)) before classify(), "
            "e.g. dspy.configure(lm=dspy.LM('openai/gpt-4.1-mini'))."
        )

    if isinstance(lm, str):
        raise HSInitializationError(
            "DSPy language model must be a dspy.LM instance, not a string. "
            f"Use dspy.configure(lm=dspy.LM('{lm}')) instead of "
            f"dspy.configure(lm='{lm}')."
        )

    if not isinstance(lm, dspy.BaseLM):
        raise HSInitializationError(
            "DSPy language model must be an instance of dspy.BaseLM, "
            f"got {type(lm).__name__}."
        )
