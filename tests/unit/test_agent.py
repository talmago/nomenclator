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
    HSClassificationSelectionModel,
    HSClassificationSelectionOutputModel,
    HSCodeCandidateModel,
)
from nomenclator.models.navigation import (
    HSResearchSelectionModel,
    HSResearchSelectionOutputModel,
)
from nomenclator.models.product_facts import MainAttributesModel, ProductFactsModel
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
    patch_candidate_chapters,
    patch_headings,
) -> None:
    """Classification should execute the full pipeline and hydrate the final result."""

    # Product Analyst
    facts = ProductFactsModel(
        raw_description="lithium ion battery pack",
        normalized_description="lithium battery pack",
        product_category="electrical equipment",
        main_attributes=MainAttributesModel(
            product_type="battery pack",
            material="lithium",
            is_part=True,
        ),
        keywords=[
            "battery",
            "battery pack",
            "lithium-ion battery",
            "electric accumulator",
            "electrical equipment",
        ],
        hints=[],
    )

    agent._product_analyst = MagicMock(
        return_value=facts,
    )

    # Retriever
    retrieved = [MagicMock()]
    research_context = MagicMock()

    agent._build_research_context = MagicMock(
        return_value=research_context,
    )

    navigation = HSResearchSelectionOutputModel(
        candidates=[
            HSResearchSelectionModel(
                chapter_ref="85",
            )
        ]
    )

    agent._research_analyst = MagicMock(return_value=navigation)

    # Chapter lookup
    chapter = chapter_mock(
        chapter_number=85,
        ref="8501-2022E",
    )

    agent._client.get_chapter.return_value = chapter

    # Headings returned by retrieval
    heading = MagicMock(
        code="8507",
        description="Electric accumulators",
        subheadings=[
            MagicMock(
                code="8507.60",
                description="Lithium-ion accumulators",
            )
        ],
    )

    # Classification Analyst
    selection = HSClassificationSelectionOutputModel(
        candidates=[
            HSClassificationSelectionModel(
                code="8507.60",
                confidence=0.9,
                reasoning=[
                    "The product is a lithium-ion battery pack.",
                ],
            )
        ]
    )

    agent._classification_analyst = MagicMock(return_value=selection)

    with (
        patch_candidate_chapters(retrieved) as retrieve,
        patch_headings({85: [heading]}),
    ):
        result = agent.classify("lithium ion battery pack")

    expected = HSCodeCandidateModel(
        code="8507.60",
        description="Electric accumulators — Lithium-ion accumulators",
        score=0.9,
        reasoning=[
            "The product is a lithium-ion battery pack.",
        ],
        source_chapter="8501-2022E",
        source_url=(
            "https://www.wcoomd.org/-/media/wco/public/global/pdf/topics/"
            "nomenclature/instruments-and-tools/hs-nomenclature-2022/2022/"
            "8501_2022e.pdf?la=en"
        ),
    )

    assert isinstance(result, HSClassificationResult)

    assert result.facts is facts
    assert result.retrieved is retrieved
    assert result.navigation is navigation

    assert result.classification.candidates == [expected]
    assert result.candidates == [expected]

    retrieve.assert_called_once_with(facts)

    agent._client.get_chapter.assert_called_once_with("85")
    agent._client.get_general_rules.assert_called_once()
    agent._classification_analyst.assert_called_once()

    facts_arg, context = agent._classification_analyst.call_args.args

    assert facts_arg is facts

    assert context.general_rules == [rule.to_dict() for rule in general_rules.rules]

    assert len(context.chapters) == 1

    chapter = context.chapters[0]
    assert chapter.chapter_number == 85
    assert chapter.title == "Electrical machinery and equipment"

    assert len(chapter.headings) == 1

    heading = chapter.headings[0]
    assert heading.code == "8507"
    assert heading.description == "Electric accumulators"

    assert len(heading.subheadings) == 1

    subheading = heading.subheadings[0]
    assert subheading.code == "8507.60"
    assert subheading.description == "Lithium-ion accumulators"


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
    agent,
    chapter_mock,
    patch_candidate_chapters,
    patch_headings,
) -> None:
    """Classification failures should be wrapped."""

    facts = MagicMock()
    facts.normalized_description = "lithium battery pack"
    facts.keywords = []

    facts.product_category = ""
    facts.main_attributes.product_type = ""
    facts.main_attributes.material = ""
    facts.main_attributes.is_part = False

    agent._product_analyst = MagicMock(
        return_value=facts,
    )

    retrieved = [MagicMock()]

    research_context = MagicMock()

    agent._build_research_context = MagicMock(
        return_value=research_context,
    )

    navigation = HSResearchSelectionOutputModel(
        candidates=[
            HSResearchSelectionModel(
                chapter_ref="85",
            )
        ]
    )

    agent._research_analyst = MagicMock(
        return_value=navigation,
    )

    chapter = chapter_mock(
        chapter_number=85,
        ref="8501-2022E",
    )

    agent._client.get_chapter.return_value = chapter

    heading = MagicMock(
        code="8507",
        description="Electric accumulators",
        subheadings=[],
    )

    agent._classification_analyst = MagicMock(
        side_effect=RuntimeError("LLM unavailable"),
    )

    with (
        patch_candidate_chapters(retrieved),
        patch_headings({85: [heading]}),
        pytest.raises(HSClassificationAnalysisError) as exc_info,
    ):
        agent.classify("lithium battery pack")

    assert "Failed to produce HS classification candidates" in str(exc_info.value)

    agent._client.get_chapter.assert_called_once_with("85")
    agent._classification_analyst.assert_called_once()


def test_agent_loads_only_selected_chapters(
    agent,
    chapter_mock,
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

    agent._product_analyst = MagicMock(
        return_value=facts,
    )

    # Pretend retrieval found many candidate chapters.
    retrieved = [MagicMock() for _ in range(5)]

    research_context = MagicMock()

    agent._build_research_context = MagicMock(
        return_value=research_context,
    )

    # Research keeps only two.
    navigation = HSResearchSelectionOutputModel(
        candidates=[
            HSResearchSelectionModel(chapter_ref="84"),
            HSResearchSelectionModel(chapter_ref="85"),
        ]
    )

    agent._research_analyst = MagicMock(
        return_value=navigation,
    )

    chapter = chapter_mock(
        ref="8501-2022E",
    )

    agent._client.get_chapter.return_value = chapter

    heading = MagicMock(
        code="8507",
        description="Electric accumulators",
        subheadings=[],
    )

    selection = HSClassificationSelectionOutputModel(
        candidates=[],
    )

    agent._classification_analyst = MagicMock(
        return_value=selection,
    )

    with (
        patch_candidate_chapters(retrieved),
        patch_headings({84: [heading], 85: [heading]}),
    ):
        result = agent.classify(
            "lithium battery pack",
        )

    assert result.classification.candidates == []

    assert agent._client.get_chapter.call_count == 2

    assert agent._client.get_chapter.call_args_list == [
        call("84"),
        call("85"),
    ]
