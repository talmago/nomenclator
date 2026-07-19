from unittest.mock import MagicMock, call, patch

import pytest

from nomenclator.agent import (
    HSClassificationAgent,
    HSClassificationAnalysisError,
    HSClassificationResult,
    HSProductAnalysisError,
    HSResearchAnalysisError,
)
from nomenclator.exceptions import (
    HSClassificationPipelineError,
    HSInitializationError,
    HSNoCandidatesFoundError,
)
from nomenclator.models.classification import (
    HSClassificationOutputModel,
    HSCodeCandidateModel,
)
from nomenclator.nomenclature.rules import HSGeneralRules


def test_classify_requires_dspy_lm() -> None:
    """classify() should fail early when DSPy LM is not configured."""

    agent = HSClassificationAgent.__new__(HSClassificationAgent)
    agent._product_analyst = MagicMock()

    with (
        patch("nomenclator.usage.dspy.settings") as settings,
        pytest.raises(HSInitializationError) as exc_info,
    ):
        settings.lm = None
        agent.classify("lithium battery pack")

    assert "language model is not configured" in str(exc_info.value)
    agent._product_analyst.assert_not_called()


def test_agent_runs_complete_pipeline(
    agent: HSClassificationAgent,
    general_rules: HSGeneralRules,
    chapter_mock,
    heading_mock,
    patch_candidate_chapters,
    patch_headings,
) -> None:
    """Classification should pass data through all pipeline stages."""

    facts = MagicMock()

    facts.normalized_description = "lithium battery pack"
    facts.keywords = ["battery"]

    facts.product_category = None

    facts.main_attributes.product_type = "battery pack"
    facts.main_attributes.material = "lithium"
    facts.main_attributes.is_part = True

    agent._product_analyst = MagicMock(
        return_value=facts,
    )

    retrieved = [MagicMock()]

    research_context = MagicMock()

    agent._build_research_context = MagicMock(
        return_value=research_context,
    )

    navigation = MagicMock()
    navigation.candidates = [
        MagicMock(chapter_ref="85"),
    ]

    agent._research_analyst = MagicMock(
        return_value=navigation,
    )

    chapter = chapter_mock(
        chapter_number=85,
        ref="8501-2022E",
    )

    agent._client.get_chapter.return_value = chapter

    heading = heading_mock({"code": "8507", "description": "Electric accumulators"})

    expected = HSClassificationOutputModel(
        candidates=[
            HSCodeCandidateModel(
                code="8507.60",
                description="Lithium-ion accumulators",
                score=0.9,
                reasoning=[
                    "The product is a lithium-ion battery pack.",
                ],
                source_chapter="85",
            )
        ]
    )

    agent._classification_analyst = MagicMock(
        return_value=expected,
    )

    with (
        patch_candidate_chapters(retrieved) as retrieve,
        patch_headings({85: [heading]}),
    ):
        result = agent.classify(
            "lithium ion battery pack",
        )

    assert result.classification is expected
    assert isinstance(result, HSClassificationResult)
    assert result.facts is facts
    assert result.retrieved is retrieved
    assert result.navigation is navigation
    assert result.candidates == expected.candidates

    retrieve.assert_called_once_with(
        "lithium battery pack",
        keywords=[
            "battery",
            "battery pack",
            "lithium",
            "part",
        ],
    )

    agent._client.get_chapter.assert_called_once_with(
        "85",
    )

    agent._classification_analyst.assert_called_once()
    agent._client.get_general_rules.assert_called_once()

    (classification_call,) = agent._classification_analyst.call_args_list
    (
        _facts_arg,
        chapter_context_arg,
        general_rules_arg,
    ) = classification_call.args

    assert chapter_context_arg == [
        {
            "chapter_number": 85,
            "title": "Electrical machinery and equipment",
            "notes": [],
            "headings": [
                {"code": "8507", "description": "Electric accumulators"},
            ],
        }
    ]
    assert general_rules_arg == [rule.to_dict() for rule in general_rules.rules]


def test_agent_raises_no_candidates(agent, patch_candidate_chapters) -> None:
    """Classification should fail when retrieval returns no candidates."""

    facts = MagicMock()

    facts.normalized_description = "unknown product"
    facts.keywords = []

    # Fields consumed by _retrieval_keywords()
    facts.product_category = ""
    facts.main_attributes.product_type = ""
    facts.main_attributes.material = ""
    facts.main_attributes.is_part = False

    agent._product_analyst.return_value = facts

    with patch_candidate_chapters([]), pytest.raises(HSNoCandidatesFoundError):
        agent.classify(
            "unknown product",
        )

    agent._research_analyst.assert_not_called()
    agent._classification_analyst.assert_not_called()


def test_agent_wraps_chapter_loading_failure(agent, patch_candidate_chapters) -> None:
    """Chapter loading failures should be wrapped as pipeline errors."""

    facts = MagicMock()

    facts.normalized_description = "lithium battery pack"
    facts.keywords = []

    # Fields consumed by _retrieval_keywords()
    facts.product_category = ""
    facts.main_attributes.product_type = ""
    facts.main_attributes.material = ""
    facts.main_attributes.is_part = False

    agent._product_analyst.return_value = facts

    research_context = MagicMock()

    agent._build_research_context = MagicMock(
        return_value=research_context,
    )

    navigation = MagicMock()
    navigation.candidates = [
        MagicMock(chapter_ref="85"),
    ]

    agent._research_analyst.return_value = navigation

    agent._client.get_chapter.side_effect = RuntimeError(
        "chapter download failed",
    )

    with (
        patch_candidate_chapters([MagicMock()]),
        pytest.raises(HSClassificationPipelineError) as exc_info,
    ):
        agent.classify(
            "lithium battery pack",
        )

    assert "Failed to load HS classification context" in str(exc_info.value)

    agent._classification_analyst.assert_not_called()
    agent._build_research_context.assert_called_once()


def test_agent_wraps_product_analysis_failure(agent, patch_candidate_chapters) -> None:
    """Product analysis failures should be wrapped."""

    agent._product_analyst.side_effect = RuntimeError(
        "LLM unavailable",
    )

    with (
        patch_candidate_chapters([]) as retrieve,
        pytest.raises(HSProductAnalysisError) as exc_info,
    ):
        agent.classify(
            "lithium battery pack",
        )

    assert "Failed to extract product facts" in str(exc_info.value)

    retrieve.assert_not_called()
    agent._research_analyst.assert_not_called()
    agent._classification_analyst.assert_not_called()


def test_agent_wraps_research_analysis_failure(agent, patch_candidate_chapters) -> None:
    """Research analysis failures should be wrapped."""

    facts = MagicMock()

    facts.normalized_description = "lithium battery pack"
    facts.keywords = []

    facts.product_category = ""
    facts.main_attributes.product_type = ""
    facts.main_attributes.material = ""
    facts.main_attributes.is_part = False

    agent._product_analyst.return_value = facts

    retrieved = [MagicMock()]

    research_context = MagicMock()

    agent._build_research_context = MagicMock(
        return_value=research_context,
    )

    agent._research_analyst.side_effect = RuntimeError(
        "LLM unavailable",
    )

    with (
        patch_candidate_chapters(retrieved),
        pytest.raises(HSResearchAnalysisError) as exc_info,
    ):
        agent.classify(
            "lithium battery pack",
        )

    assert "Failed to analyze retrieved HS candidates" in str(exc_info.value)

    agent._build_research_context.assert_called_once_with(
        retrieved,
    )
    agent._classification_analyst.assert_not_called()
    agent._client.get_chapter.assert_not_called()


def test_agent_wraps_classification_failure(
    agent, chapter_mock, heading_mock, patch_candidate_chapters, patch_headings
) -> None:
    """Classification failures should be wrapped."""

    facts = MagicMock()

    facts.normalized_description = "lithium battery pack"
    facts.keywords = []

    facts.product_category = ""
    facts.main_attributes.product_type = ""
    facts.main_attributes.material = ""
    facts.main_attributes.is_part = False

    agent._product_analyst.return_value = facts

    retrieved = [MagicMock()]

    research_context = MagicMock()

    agent._build_research_context = MagicMock(
        return_value=research_context,
    )

    navigation = MagicMock()
    navigation.candidates = [
        MagicMock(chapter_ref="85"),
    ]

    agent._research_analyst.return_value = navigation

    chapter = chapter_mock(
        chapter_number=85,
        ref="8501-2022E",
    )

    agent._client.get_chapter.return_value = chapter

    agent._classification_analyst.side_effect = RuntimeError(
        "LLM unavailable",
    )

    with (
        patch_candidate_chapters(retrieved),
        patch_headings({85: [heading_mock({"chapter_number": 85})]}),
        pytest.raises(HSClassificationAnalysisError) as exc_info,
    ):
        agent.classify(
            "lithium battery pack",
        )

    assert "Failed to produce HS classification candidates" in str(exc_info.value)

    agent._client.get_chapter.assert_called_once_with("85")
    agent._classification_analyst.assert_called_once()


def test_agent_loads_only_selected_chapters(
    agent,
    chapter_mock,
    heading_mock,
    patch_candidate_chapters,
    patch_headings,
) -> None:
    """Only chapters selected by the Research Analyst should be loaded."""

    facts = MagicMock()

    facts.normalized_description = "lithium battery pack"
    facts.keywords = []

    facts.product_category = ""
    facts.main_attributes.product_type = ""
    facts.main_attributes.material = ""
    facts.main_attributes.is_part = False

    agent._product_analyst.return_value = facts

    # Pretend retrieval found many candidate chapters.
    retrieved = [MagicMock() for _ in range(5)]

    research_context = MagicMock()

    agent._build_research_context = MagicMock(
        return_value=research_context,
    )

    # Research keeps only two.
    navigation = MagicMock()
    navigation.candidates = [
        MagicMock(chapter_ref="84"),
        MagicMock(chapter_ref="85"),
    ]

    agent._research_analyst.return_value = navigation

    chapter = chapter_mock(
        ref="8501-2022E",
    )

    agent._client.get_chapter.return_value = chapter

    expected = HSClassificationOutputModel(
        candidates=[],
    )

    agent._classification_analyst.return_value = expected

    with (
        patch_candidate_chapters(retrieved),
        patch_headings({85: [heading_mock({})]}),
    ):
        result = agent.classify(
            "lithium battery pack",
        )

    assert result.classification is expected

    assert agent._client.get_chapter.call_count == 2

    assert agent._client.get_chapter.call_args_list == [
        call("84"),
        call("85"),
    ]
