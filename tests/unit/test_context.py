from unittest.mock import MagicMock, patch

from nomenclator.agent import HSClassificationAgent
from nomenclator.models.classification import HSClassificationOutputModel
from nomenclator.nomenclature.rules import HSGeneralRules
from nomenclator.nomenclature.tree import HSDocumentRef, HSSection


def _heading_mock(heading_dict: dict) -> MagicMock:
    """Build a mocked HSHeading whose to_dict returns ``heading_dict``."""

    heading = MagicMock()
    heading.to_dict.return_value = heading_dict
    return heading


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
