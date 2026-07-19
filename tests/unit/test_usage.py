from unittest.mock import patch

import dspy
import pytest

from nomenclator.exceptions import HSInitializationError
from nomenclator.usage import calc_usage, ensure_dspy_lm


def test_ensure_dspy_lm_requires_configured_lm() -> None:
    """Missing DSPy LM configuration should raise HSInitializationError."""

    with (
        patch("nomenclator.usage.dspy.settings") as settings,
        pytest.raises(HSInitializationError) as exc_info,
    ):
        settings.lm = None
        ensure_dspy_lm()

    assert "language model is not configured" in str(exc_info.value)


def test_ensure_dspy_lm_rejects_string_lm() -> None:
    """A bare model string must not be accepted as a configured LM."""

    with (
        patch("nomenclator.usage.dspy.settings") as settings,
        pytest.raises(HSInitializationError) as exc_info,
    ):
        settings.lm = "openai/gpt-4.1-mini"
        ensure_dspy_lm()

    assert "must be a dspy.LM instance, not a string" in str(exc_info.value)


def test_ensure_dspy_lm_rejects_non_base_lm() -> None:
    """Non-BaseLM values must not be accepted as a configured LM."""

    with (
        patch("nomenclator.usage.dspy.settings") as settings,
        pytest.raises(HSInitializationError) as exc_info,
    ):
        settings.lm = object()
        ensure_dspy_lm()

    assert "must be an instance of dspy.BaseLM" in str(exc_info.value)


def test_ensure_dspy_lm_accepts_base_lm() -> None:
    """A configured BaseLM instance should pass the initialization check."""

    class _FakeLM(dspy.BaseLM):
        def __init__(self) -> None:
            pass

        def __call__(self, *args, **kwargs):
            return []

    with patch("nomenclator.usage.dspy.settings") as settings:
        settings.lm = _FakeLM()
        ensure_dspy_lm()


def test_calc_usage_sums_token_usage() -> None:
    """Token usage should be aggregated across LM history entries."""

    history = [
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
            }
        },
        {
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 10,
            }
        },
    ]

    usage = calc_usage(history)

    assert usage.prompt_tokens == 150
    assert usage.completion_tokens == 30


def test_calc_usage_ignores_missing_usage() -> None:
    """History entries without usage should not affect totals."""

    history = [
        {},
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
            }
        },
    ]

    usage = calc_usage(history)

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5


def test_calc_usage_defaults_cost_to_zero() -> None:
    """Missing cost information should default to zero."""

    history = [
        {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            }
        },
    ]

    usage = calc_usage(history)

    assert usage.prompt_tokens == 1000
    assert usage.completion_tokens == 500
    assert usage.total_tokens == 1500
    assert usage.cost == 0.0


def test_calc_usage_preserves_cost_from_payload() -> None:
    """Provided cost information should be preserved."""

    history = [
        {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            },
            "cost": 0.0025,
        },
    ]

    usage = calc_usage(history)

    assert usage.prompt_tokens == 1000
    assert usage.completion_tokens == 500
    assert usage.total_tokens == 1500
    assert usage.cost == 0.0025
