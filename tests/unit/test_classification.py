from unittest.mock import MagicMock, call, patch

import pytest

from nomenclator.agent import (
    HSClassificationAgent,
    HSClassificationAnalysisError,
    HSProductAnalysisError,
    HSResearchAnalysisError,
)
from nomenclator.exceptions import (
    HSClassificationPipelineError,
    HSNoCandidatesFoundError,
)
from nomenclator.models.classification import (
    HSClassificationOutputModel,
    HSCodeCandidateModel,
)
from nomenclator.models.tree import HSDocumentRef, HSSection
from nomenclator.usage import calc_usage


@pytest.fixture
def agent() -> HSClassificationAgent:
    """Create an agent with mocked dependencies."""

    client = MagicMock()

    agent = HSClassificationAgent.__new__(HSClassificationAgent)

    agent._client = client
    agent._retriever = MagicMock()
    agent._retrieval_limit = 5

    agent._product_analyst = MagicMock()
    agent._research_analyst = MagicMock()
    agent._classification_analyst = MagicMock()

    return agent


def test_agent_pipeline(
    agent: HSClassificationAgent,
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
    agent._retriever.search.return_value = retrieved

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

    chapter_context = MagicMock()

    chapter = MagicMock()
    chapter.to_classification_context.return_value = chapter_context

    agent._client.get_chapter.return_value = chapter

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

    result = agent.classify(
        "lithium ion battery pack",
    )

    assert result is expected

    agent._retriever.search.assert_called_once_with(
        "lithium battery pack",
        keywords=[
            "battery",
            "battery pack",
            "lithium",
            "part",
        ],
        limit=agent._retrieval_limit,
    )

    agent._client.get_chapter.assert_called_once_with(
        "85",
    )

    agent._classification_analyst.assert_called_once()


def test_agent_raises_no_candidates(
    agent: HSClassificationAgent,
) -> None:
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

    agent._retriever.search.return_value = []

    with pytest.raises(HSNoCandidatesFoundError):
        agent.classify(
            "unknown product",
        )

    agent._research_analyst.assert_not_called()
    agent._classification_analyst.assert_not_called()


def test_build_research_context_groups_chapters_by_section(
    agent: HSClassificationAgent,
) -> None:
    """Retrieved chapters from the same section should share one section context."""

    chapter_84 = HSDocumentRef(
        title="Nuclear reactors, boilers, machinery...",
        ref="8401-2022E",
    )

    chapter_85 = HSDocumentRef(
        title="Electrical machinery and equipment...",
        ref="8501-2022E",
    )

    result_84 = MagicMock()
    result_84.document.payload = chapter_84
    result_84.score = 0.92

    result_85 = MagicMock()
    result_85.document.payload = chapter_85
    result_85.score = 0.88

    section = HSSection(
        id="section_xvi",
        label="SECTION XVI",
        title="MACHINERY AND MECHANICAL APPLIANCES...",
    )

    section_notes = MagicMock()
    section_notes.to_dict.return_value = {
        "notes": [],
    }

    agent._client.get_section_for_chapter = MagicMock(
        return_value=section,
    )

    agent._client.get_section_notes = MagicMock(
        return_value=section_notes,
    )

    context = agent._build_research_context(
        [result_84, result_85],
    )

    assert len(context.sections) == 1

    research_section = context.sections[0]

    assert research_section.section_id == "section_xvi"
    assert research_section.section_title == section.title
    assert research_section.section_notes == {"notes": []}

    assert len(research_section.chapters) == 2

    assert research_section.chapters[0].chapter_ref == "8401-2022E"
    assert research_section.chapters[1].chapter_ref == "8501-2022E"

    agent._client.get_section_notes.assert_called_once_with(
        "section_xvi",
    )


def test_build_research_context_multiple_sections(
    agent: HSClassificationAgent,
) -> None:
    """Retrieved chapters from different sections should produce separate contexts."""

    chapter_85 = HSDocumentRef(
        title="Electrical machinery and equipment",
        ref="8501-2022E",
    )

    chapter_39 = HSDocumentRef(
        title="Plastics and articles thereof",
        ref="3901-2022E",
    )

    result_85 = MagicMock()
    result_85.document.payload = chapter_85
    result_85.score = 0.95

    result_39 = MagicMock()
    result_39.document.payload = chapter_39
    result_39.score = 0.81

    section_xvi = HSSection(
        id="section_xvi",
        label="SECTION XVI",
        title="Machinery and mechanical appliances",
    )

    section_vii = HSSection(
        id="section_vii",
        label="SECTION VII",
        title="Plastics and articles thereof",
    )

    notes_xvi = MagicMock()
    notes_xvi.to_dict.return_value = {"notes": ["Section XVI note"]}

    notes_vii = MagicMock()
    notes_vii.to_dict.return_value = {"notes": ["Section VII note"]}

    agent._client.get_section_for_chapter = MagicMock(
        side_effect=[
            section_xvi,
            section_vii,
        ],
    )

    agent._client.get_section_notes = MagicMock(
        side_effect=[
            notes_xvi,
            notes_vii,
        ],
    )

    context = agent._build_research_context(
        [result_85, result_39],
    )

    assert len(context.sections) == 2

    first = context.sections[0]
    second = context.sections[1]

    assert first.section_id == "section_xvi"
    assert first.chapters[0].chapter_ref == "8501-2022E"
    assert first.section_notes == {"notes": ["Section XVI note"]}

    assert second.section_id == "section_vii"
    assert second.chapters[0].chapter_ref == "3901-2022E"
    assert second.section_notes == {"notes": ["Section VII note"]}

    assert agent._client.get_section_notes.call_count == 2
    agent._client.get_section_notes.assert_any_call("section_xvi")
    agent._client.get_section_notes.assert_any_call("section_vii")


def test_agent_wraps_chapter_loading_failure(
    agent: HSClassificationAgent,
) -> None:
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

    agent._retriever.search.return_value = [
        MagicMock(),
    ]

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

    with pytest.raises(HSClassificationPipelineError) as exc_info:
        agent.classify(
            "lithium battery pack",
        )

    assert "Failed to load HS chapter context" in str(exc_info.value)

    agent._classification_analyst.assert_not_called()
    agent._build_research_context.assert_called_once()


def test_agent_wraps_product_analysis_failure(
    agent: HSClassificationAgent,
) -> None:
    """Product analysis failures should be wrapped."""

    agent._product_analyst.side_effect = RuntimeError(
        "LLM unavailable",
    )

    with pytest.raises(HSProductAnalysisError) as exc_info:
        agent.classify(
            "lithium battery pack",
        )

    assert "Failed to extract product facts" in str(exc_info.value)

    agent._retriever.search.assert_not_called()
    agent._research_analyst.assert_not_called()
    agent._classification_analyst.assert_not_called()


def test_agent_wraps_research_analysis_failure(
    agent: HSClassificationAgent,
) -> None:
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

    agent._retriever.search.return_value = retrieved

    research_context = MagicMock()

    agent._build_research_context = MagicMock(
        return_value=research_context,
    )

    agent._research_analyst.side_effect = RuntimeError(
        "LLM unavailable",
    )

    with pytest.raises(HSResearchAnalysisError) as exc_info:
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
    agent: HSClassificationAgent,
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

    agent._retriever.search.return_value = retrieved

    research_context = MagicMock()

    agent._build_research_context = MagicMock(
        return_value=research_context,
    )

    navigation = MagicMock()
    navigation.candidates = [
        MagicMock(chapter_ref="85"),
    ]

    agent._research_analyst.return_value = navigation

    chapter = MagicMock()
    chapter.to_classification_context.return_value = {
        "chapter_number": 85,
    }

    agent._client.get_chapter.return_value = chapter

    agent._classification_analyst.side_effect = RuntimeError(
        "LLM unavailable",
    )

    with pytest.raises(HSClassificationAnalysisError) as exc_info:
        agent.classify(
            "lithium battery pack",
        )

    assert "Failed to produce HS classification candidates" in str(exc_info.value)

    agent._client.get_chapter.assert_called_once_with("85")
    agent._classification_analyst.assert_called_once()


def test_retrieval_keywords_builds_unique_keywords() -> None:
    """Retrieval keywords should combine facts and hints."""

    facts = MagicMock()

    facts.keywords = [
        "cotton",
        "shirt",
        "cotton",  # duplicate
    ]

    facts.product_category = "apparel"

    facts.main_attributes.product_type = "knitted shirt"
    facts.main_attributes.material = "cotton"
    facts.main_attributes.is_part = False

    agent = HSClassificationAgent.__new__(
        HSClassificationAgent,
    )

    keywords = agent._retrieval_keywords(
        facts,
        user_hs_codes=["6105"],
    )

    assert keywords == [
        "cotton",
        "shirt",
        "apparel",
        "knitted shirt",
        "6105",
    ]


def test_retrieval_keywords_adds_part_keyword() -> None:
    """Part products should add a part retrieval hint."""

    facts = MagicMock()

    facts.keywords = []
    facts.product_category = None

    facts.main_attributes.product_type = "battery pack"
    facts.main_attributes.material = "lithium"
    facts.main_attributes.is_part = True

    agent = HSClassificationAgent.__new__(
        HSClassificationAgent,
    )

    keywords = agent._retrieval_keywords(
        facts,
        None,
    )

    assert keywords == [
        "battery pack",
        "lithium",
        "part",
    ]


def test_build_hs_navigation_retriever_builds_documents() -> None:
    """HS navigation retriever should be built from tree chapters."""

    agent = HSClassificationAgent.__new__(
        HSClassificationAgent,
    )

    chapter = MagicMock(
        ref="0101-2022E",
        title="Live animals.",
    )

    section = MagicMock(
        label="SECTION I",
        title="LIVE ANIMALS; ANIMAL PRODUCTS",
        chapters=[chapter],
    )

    agent._tree = MagicMock(
        sections=[section],
    )

    with patch(
        "nomenclator.agent.Retriever",
    ) as retriever_cls:
        retriever = agent._build_hs_navigation_retriever(
            model_name="test-model",
        )

    retriever_cls.assert_called_once()

    (documents,) = retriever_cls.call_args.args

    assert len(documents) == 1

    document = documents[0]

    assert document.id == "0101-2022E"
    assert document.payload == chapter
    assert "SECTION I" in document.content
    assert "LIVE ANIMALS" in document.content

    assert retriever is retriever_cls.return_value


def test_build_hs_navigation_retriever_skips_chapters_without_refs() -> None:
    agent = HSClassificationAgent.__new__(HSClassificationAgent)

    valid = MagicMock(ref="0101-2022E")
    invalid = MagicMock(ref=None)

    section = MagicMock(
        label="SECTION I",
        title="LIVE ANIMALS",
        chapters=[valid, invalid],
    )

    agent._tree = MagicMock(
        sections=[section],
    )

    with patch("nomenclator.agent.Retriever") as retriever_cls:
        agent._build_hs_navigation_retriever(
            model_name=None,
        )

    (documents,) = retriever_cls.call_args.args

    assert [doc.id for doc in documents] == [
        "0101-2022E",
    ]


def test_agent_loads_only_selected_chapters(
    agent: HSClassificationAgent,
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

    agent._retriever.search.return_value = retrieved

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

    chapter = MagicMock()
    chapter.to_classification_context.return_value = {}

    agent._client.get_chapter.return_value = chapter

    expected = HSClassificationOutputModel(
        candidates=[],
    )

    agent._classification_analyst.return_value = expected

    result = agent.classify(
        "lithium battery pack",
    )

    assert result is expected

    assert agent._client.get_chapter.call_count == 2

    assert agent._client.get_chapter.call_args_list == [
        call("84"),
        call("85"),
    ]


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
