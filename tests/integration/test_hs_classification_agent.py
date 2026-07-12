import dspy
import pytest

from nomenclator.agent import HSClassificationAgent
from nomenclator.usage import calc_usage


@pytest.mark.integration
def test_hs_classification_agent() -> None:
    """Run an end-to-end HS classification smoke test.

    This test requires an OpenAI API key and executes the full pipeline:
    Product Analyst -> Retriever -> Research Analyst -> Classification Analyst.
    """

    lm = dspy.LM(
        "openai/gpt-4.1-mini",
        cache=False,
    )

    dspy.configure(
        lm=lm,
    )

    agent = HSClassificationAgent()

    result = agent.classify(
        "Lithium ion battery pack for electric vehicles",
    )

    assert result.candidates

    top_candidate = result.candidates[0]

    assert top_candidate.code
    assert top_candidate.description
    assert top_candidate.reasoning

    print()
    print("TOP CANDIDATE")
    print("-" * 80)
    print(f"Code: {top_candidate.code}")
    print(f"Description: {top_candidate.description}")
    print(f"Score: {top_candidate.score}")
    print("Reasoning:")

    for reason in top_candidate.reasoning:
        print(f" - {reason}")

    print()
    print("Usage:")
    print(calc_usage(lm.history))
