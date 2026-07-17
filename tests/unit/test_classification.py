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
from nomenclator.nomenclature.rules import HSGeneralRule, HSGeneralRules
from nomenclator.nomenclature.tree import HSDocumentRef, HSSection
from nomenclator.usage import calc_usage, ensure_dspy_lm


@pytest.fixture
def general_rules() -> HSGeneralRules:
    """Compact fake GIR payload for classification pipeline tests."""

    return HSGeneralRules(
        title="General Rules for the interpretation of the Harmonized System",
        document=HSDocumentRef(
            title="General Rules",
            ref="0001-2202E GIR",
        ),
        rules=[
            HSGeneralRule(
                rule="1",
                text=(
                    "Classification shall be determined according to the "
                    "terms of the headings."
                ),
            ),
            HSGeneralRule(
                rule="2(a)",
                text=(
                    "Any reference to an article includes incomplete or "
                    "unfinished articles."
                ),
            ),
        ],
    )


@pytest.fixture
def agent(general_rules: HSGeneralRules) -> HSClassificationAgent:
    """Create an agent with mocked dependencies."""

    client = MagicMock()
    client.get_general_rules.return_value = general_rules

    agent = HSClassificationAgent.__new__(HSClassificationAgent)

    agent._client = client
    agent._retrieval_limit = 5
    agent._model_name = "test-model"
    agent._max_candidates = 3
    agent._max_chunks = 20

    agent._product_analyst = MagicMock()
    agent._research_analyst = MagicMock()
    agent._classification_analyst = MagicMock()

    with patch("nomenclator.agent.ensure_dspy_lm"):
        yield agent


def _mock_chapter(
    *,
    chapter_number: int = 85,
    title: str = "Electrical machinery and equipment",
    ref: str = "8501-2022E",
    notes: list | None = None,
) -> MagicMock:
    """Build a mocked HSChapter."""

    chapter = MagicMock()
    chapter.chapter_number = chapter_number
    chapter.title = title
    chapter.document.ref = ref
    chapter.notes = notes if notes is not None else []
    chapter.headings = []
    return chapter


def _heading_mock(heading_dict: dict) -> MagicMock:
    """Build a mocked HSHeading whose to_dict returns ``heading_dict``."""

    heading = MagicMock()
    heading.to_dict.return_value = heading_dict
    return heading


def _patch_candidate_chapters(retrieved: list):
    """Patch ``_retrieve_chapters`` to return ``retrieved``.

    Returns the patch context manager so callers can assert on the call
    (description and keywords).
    """

    return patch.object(
        HSClassificationAgent,
        "_retrieve_chapters",
        return_value=retrieved,
    )


def _patch_headings(
    headings_by_chapter: dict[int, list[MagicMock]],
):
    """Patch ``_retrieve_headings`` to return a fixed mapping.

    The mapping keys are chapter numbers and values are the heading mocks that
    should appear in that chapter's classification context.
    """

    return patch.object(
        HSClassificationAgent,
        "_retrieve_headings",
        return_value=headings_by_chapter,
    )


def test_ensure_dspy_lm_requires_configured_lm() -> None:
    """Missing DSPy LM configuration should raise HSInitializationError."""

    with (
        patch("nomenclator.agent.dspy.settings") as settings,
        pytest.raises(HSInitializationError) as exc_info,
    ):
        settings.lm = None
        ensure_dspy_lm()

    assert "language model is not configured" in str(exc_info.value)


def test_ensure_dspy_lm_rejects_string_lm() -> None:
    """A bare model string must not be accepted as a configured LM."""

    with (
        patch("nomenclator.agent.dspy.settings") as settings,
        pytest.raises(HSInitializationError) as exc_info,
    ):
        settings.lm = "openai/gpt-4.1-mini"
        ensure_dspy_lm()

    assert "must be a dspy.LM instance, not a string" in str(exc_info.value)


def test_ensure_dspy_lm_rejects_non_base_lm() -> None:
    """Non-BaseLM values must not be accepted as a configured LM."""

    with (
        patch("nomenclator.agent.dspy.settings") as settings,
        pytest.raises(HSInitializationError) as exc_info,
    ):
        settings.lm = object()
        ensure_dspy_lm()

    assert "must be an instance of dspy.BaseLM" in str(exc_info.value)


def test_ensure_dspy_lm_accepts_base_lm() -> None:
    """A configured BaseLM instance should pass the initialization check."""

    import dspy

    class _FakeLM(dspy.BaseLM):
        def __init__(self) -> None:
            pass

        def __call__(self, *args, **kwargs):
            return []

    with patch("nomenclator.agent.dspy.settings") as settings:
        settings.lm = _FakeLM()
        ensure_dspy_lm()


def test_classify_requires_dspy_lm() -> None:
    """classify() should fail early when DSPy LM is not configured."""

    agent = HSClassificationAgent.__new__(HSClassificationAgent)
    agent._product_analyst = MagicMock()

    with (
        patch("nomenclator.agent.dspy.settings") as settings,
        pytest.raises(HSInitializationError) as exc_info,
    ):
        settings.lm = None
        agent.classify("lithium battery pack")

    assert "language model is not configured" in str(exc_info.value)
    agent._product_analyst.assert_not_called()


def test_agent_pipeline(
    agent: HSClassificationAgent,
    general_rules: HSGeneralRules,
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

    chapter = _mock_chapter(
        chapter_number=85,
        ref="8501-2022E",
    )

    agent._client.get_chapter.return_value = chapter

    heading = _heading_mock({"code": "8507", "description": "Electric accumulators"})

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
        _patch_candidate_chapters(retrieved) as retrieve,
        _patch_headings({85: [heading]}),
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
        max_candidates_arg,
    ) = classification_call.args

    assert max_candidates_arg == agent._max_candidates

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

    with _patch_candidate_chapters([]), pytest.raises(HSNoCandidatesFoundError):
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
        _patch_candidate_chapters([MagicMock()]),
        pytest.raises(HSClassificationPipelineError) as exc_info,
    ):
        agent.classify(
            "lithium battery pack",
        )

    assert "Failed to load HS classification context" in str(exc_info.value)

    agent._classification_analyst.assert_not_called()
    agent._build_research_context.assert_called_once()


def test_agent_wraps_product_analysis_failure(
    agent: HSClassificationAgent,
) -> None:
    """Product analysis failures should be wrapped."""

    agent._product_analyst.side_effect = RuntimeError(
        "LLM unavailable",
    )

    with (
        _patch_candidate_chapters([]) as retrieve,
        pytest.raises(HSProductAnalysisError) as exc_info,
    ):
        agent.classify(
            "lithium battery pack",
        )

    assert "Failed to extract product facts" in str(exc_info.value)

    retrieve.assert_not_called()
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

    research_context = MagicMock()

    agent._build_research_context = MagicMock(
        return_value=research_context,
    )

    agent._research_analyst.side_effect = RuntimeError(
        "LLM unavailable",
    )

    with (
        _patch_candidate_chapters(retrieved),
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

    research_context = MagicMock()

    agent._build_research_context = MagicMock(
        return_value=research_context,
    )

    navigation = MagicMock()
    navigation.candidates = [
        MagicMock(chapter_ref="85"),
    ]

    agent._research_analyst.return_value = navigation

    chapter = _mock_chapter(
        chapter_number=85,
        ref="8501-2022E",
    )

    agent._client.get_chapter.return_value = chapter

    agent._classification_analyst.side_effect = RuntimeError(
        "LLM unavailable",
    )

    with (
        _patch_candidate_chapters(retrieved),
        _patch_headings({85: [_heading_mock({"chapter_number": 85})]}),
        pytest.raises(HSClassificationAnalysisError) as exc_info,
    ):
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


def test_retrieve_candidate_chapters_builds_documents() -> None:
    """Candidate chapter retrieval should build documents from tree chapters."""

    agent = HSClassificationAgent.__new__(
        HSClassificationAgent,
    )
    agent._model_name = "test-model"
    agent._retrieval_limit = 5

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
        results = agent._retrieve_chapters(
            "live horses",
            keywords=["horse", "animal"],
        )

    retriever_cls.assert_called_once()

    (documents,) = retriever_cls.call_args.args
    (model_name_kwarg,) = retriever_cls.call_args.kwargs.values()
    assert model_name_kwarg == "test-model"

    assert len(documents) == 1

    document = documents[0]

    assert document.id == "0101-2022E"
    assert document.payload == chapter
    assert "SECTION I" in document.content
    assert "LIVE ANIMALS" in document.content

    retriever_cls.return_value.search.assert_called_once_with(
        "live horses",
        keywords=["horse", "animal"],
        limit=5,
    )
    assert results is retriever_cls.return_value.search.return_value


def test_retrieve_chapters_skips_chapters_without_refs() -> None:
    """Chapters without a reference should be excluded from the retriever."""

    agent = HSClassificationAgent.__new__(HSClassificationAgent)
    agent._model_name = "test-model"
    agent._retrieval_limit = 5

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
        agent._retrieve_chapters(
            "live horses",
            keywords=[],
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

    chapter = _mock_chapter(
        ref="8501-2022E",
    )

    agent._client.get_chapter.return_value = chapter

    expected = HSClassificationOutputModel(
        candidates=[],
    )

    agent._classification_analyst.return_value = expected

    with (
        _patch_candidate_chapters(retrieved),
        _patch_headings({85: [_heading_mock({})]}),
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


def test_heading_chunks_produces_heading_chunks() -> None:
    """Each heading should become one retrieval chunk with heading as payload."""

    from nomenclator.nomenclature.chapter import HSChapter
    from nomenclator.nomenclature.tree import HSDocumentRef, HSHeading, HSSubheading

    heading_a = HSHeading(
        code="85.07",
        description="Electric accumulators",
        subheadings=[
            HSSubheading(code="8507.10", description="Lead-acid accumulators"),
            HSSubheading(code="8507.60", description="Lithium-ion accumulators"),
        ],
    )

    heading_b = HSHeading(
        code="85.08",
        description="Electrical vacuum cleaners",
        subheadings=[],
    )

    chapter = HSChapter(
        chapter_number=85,
        title="Electrical machinery and equipment",
        document=HSDocumentRef(title="Chapter 85", ref="8501-2022E"),
        notes=[],
        headings=[heading_a, heading_b],
    )

    chunks = HSClassificationAgent._split_chapter_into_chunks(chapter)

    assert len(chunks) == 2

    assert chunks[0].id == "85:85.07"
    assert chunks[0].payload is heading_a
    assert "Heading 85.07" in chunks[0].content
    assert "Electric accumulators" in chunks[0].content
    assert "8507.10" in chunks[0].content
    assert "Lead-acid accumulators" in chunks[0].content
    assert "8507.60" in chunks[0].content
    assert "Lithium-ion accumulators" in chunks[0].content

    assert chunks[1].id == "85:85.08"
    assert chunks[1].payload is heading_b
    assert "Heading 85.08" in chunks[1].content
    assert "Electrical vacuum cleaners" in chunks[1].content


def test_heading_chunk_content_includes_group_path() -> None:
    """Subheading group labels should be rendered into chunk content."""

    from nomenclator.nomenclature.chapter import HSChapter
    from nomenclator.nomenclature.tree import HSDocumentRef, HSHeading, HSSubheading

    heading = HSHeading(
        code="85.07",
        description="Electric accumulators",
        subheadings=[
            HSSubheading(
                code="8507.80",
                description="Other accumulators",
                group_path=["Other"],
            ),
        ],
    )

    chapter = HSChapter(
        chapter_number=85,
        title="Electrical machinery",
        document=HSDocumentRef(title="Chapter 85", ref="8501-2022E"),
        headings=[heading],
    )

    (chunk,) = HSClassificationAgent._split_chapter_into_chunks(chapter)

    assert "Other" in chunk.content
    assert "8507.80" in chunk.content


def test_classification_context_always_includes_chapter_notes(
    agent: HSClassificationAgent,
    general_rules: HSGeneralRules,
) -> None:
    """Chapter notes should be included regardless of retrieved chunks."""

    facts = MagicMock()
    facts.normalized_description = "lithium battery pack"
    facts.keywords = []

    facts.product_category = ""
    facts.main_attributes.product_type = ""
    facts.main_attributes.material = ""
    facts.main_attributes.is_part = False

    agent._product_analyst.return_value = facts

    agent._build_research_context = MagicMock(return_value=MagicMock())

    navigation = MagicMock()
    navigation.candidates = [MagicMock(chapter_ref="85")]

    agent._research_analyst.return_value = navigation

    note = MagicMock()
    note.to_dict.return_value = {"number": "1", "intro": "Chapter note"}

    chapter = _mock_chapter(
        chapter_number=85,
        ref="8501-2022E",
        notes=[note],
    )

    agent._client.get_chapter.return_value = chapter

    heading = _heading_mock({"code": "8507", "description": "Electric accumulators"})

    agent._classification_analyst.return_value = HSClassificationOutputModel(
        candidates=[],
    )

    with _patch_candidate_chapters([MagicMock()]), _patch_headings({85: [heading]}):
        agent.classify("lithium battery pack")

    (
        _,
        chapter_context_arg,
        general_rules_arg,
        _,
    ) = agent._classification_analyst.call_args.args

    assert chapter_context_arg == [
        {
            "chapter_number": 85,
            "title": "Electrical machinery and equipment",
            "notes": [{"number": "1", "intro": "Chapter note"}],
            "headings": [
                {"code": "8507", "description": "Electric accumulators"},
            ],
        }
    ]
    assert general_rules_arg == [rule.to_dict() for rule in general_rules.rules]


def test_classification_context_handles_chapter_without_headings(
    agent: HSClassificationAgent,
    general_rules: HSGeneralRules,
) -> None:
    """A chapter with no headings should yield context with empty headings."""

    facts = MagicMock()
    facts.normalized_description = "rare product"
    facts.keywords = []

    facts.product_category = ""
    facts.main_attributes.product_type = ""
    facts.main_attributes.material = ""
    facts.main_attributes.is_part = False

    agent._product_analyst.return_value = facts

    agent._build_research_context = MagicMock(return_value=MagicMock())

    navigation = MagicMock()
    navigation.candidates = [MagicMock(chapter_ref="99")]

    agent._research_analyst.return_value = navigation

    note = MagicMock()
    note.to_dict.return_value = {"number": "1", "intro": "Note"}

    chapter = _mock_chapter(
        chapter_number=99,
        ref="9901-2022E",
        notes=[note],
    )

    agent._client.get_chapter.return_value = chapter

    agent._classification_analyst.return_value = HSClassificationOutputModel(
        candidates=[],
    )

    # No chunks for chapter 99, so the retriever is never built.
    with (
        _patch_candidate_chapters([MagicMock()]),
        patch.object(
            HSClassificationAgent,
            "_retrieve_headings",
            return_value={},
        ) as retrieve_headings,
    ):
        agent.classify("rare product")

    retrieve_headings.assert_called_once()

    (
        _,
        chapter_context_arg,
        general_rules_arg,
        _,
    ) = agent._classification_analyst.call_args.args

    assert chapter_context_arg == [
        {
            "chapter_number": 99,
            "title": "Electrical machinery and equipment",
            "notes": [{"number": "1", "intro": "Note"}],
            "headings": [],
        }
    ]
    assert general_rules_arg == [rule.to_dict() for rule in general_rules.rules]


def _search_result(chunk_id: str, chapter_number: int, heading_dict: dict):
    """Build a fake SearchResult for ``_select_headings`` tests."""

    heading = _heading_mock(heading_dict)

    document = MagicMock()
    document.id = chunk_id
    document.payload = heading

    result = MagicMock()
    result.document = document
    return result


def test_select_headings_guarantees_one_per_chapter(
    agent: HSClassificationAgent,
) -> None:
    """Every chapter with chunks must get at least one heading (floor)."""

    # Global ranking puts all of chapter 90 first; chapter 85's only chunk
    # is ranked last. The floor must still give chapter 85 one heading.
    results = [
        _search_result("90:9001", 90, {"code": "9001"}),
        _search_result("90:9002", 90, {"code": "9002"}),
        _search_result("90:9003", 90, {"code": "9003"}),
        _search_result("85:8507", 85, {"code": "8507"}),
    ]

    selected = HSClassificationAgent._select_headings(
        results,
        chunk_to_chapter={
            "90:9001": 90,
            "90:9002": 90,
            "90:9003": 90,
            "85:8507": 85,
        },
        chapters_with_chunks={85, 90},
        budget=3,
    )

    # Budget 3: one for 85 (floor), then two for 90.
    assert len(selected[85]) == 1
    assert selected[85][0].to_dict() == {"code": "8507"}
    assert len(selected[90]) == 2
    assert [h.to_dict() for h in selected[90]] == [{"code": "9001"}, {"code": "9002"}]


def test_select_headings_respects_budget(
    agent: HSClassificationAgent,
) -> None:
    """Total selected headings must never exceed the global budget."""

    results = [
        _search_result(f"85:850{i}", 85, {"code": f"850{i}"}) for i in range(1, 6)
    ]

    selected = HSClassificationAgent._select_headings(
        results,
        chunk_to_chapter={f"85:850{i}": 85 for i in range(1, 6)},
        chapters_with_chunks={85},
        budget=3,
    )

    total = sum(len(headings) for headings in selected.values())
    assert total == 3
    assert [h.to_dict() for h in selected[85]] == [
        {"code": "8501"},
        {"code": "8502"},
        {"code": "8503"},
    ]


def test_select_headings_budget_below_chapter_count(
    agent: HSClassificationAgent,
) -> None:
    """When the budget is smaller than the number of chapters, the floor phase
    stops at the budget, so only the highest-ranked chapters are represented."""

    results = [
        _search_result("90:9001", 90, {"code": "9001"}),
        _search_result("85:8507", 85, {"code": "8507"}),
        _search_result("84:8401", 84, {"code": "8401"}),
    ]

    selected = HSClassificationAgent._select_headings(
        results,
        chunk_to_chapter={
            "90:9001": 90,
            "85:8507": 85,
            "84:8401": 84,
        },
        chapters_with_chunks={84, 85, 90},
        budget=2,
    )

    total = sum(len(headings) for headings in selected.values())
    assert total == 2
    assert selected[90] and len(selected[90]) == 1
    assert selected[85] and len(selected[85]) == 1
    assert selected[84] == []
