from nomenclator.models.usage import TokenUsage


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
