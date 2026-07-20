from unittest.mock import MagicMock, patch

from nomenclator.agent import HSClassificationAgent
from nomenclator.models.classification import (
    HSClassificationSelectionOutputModel,
)
from nomenclator.models.navigation import (
    HSResearchSelectionModel,
    HSResearchSelectionOutputModel,
)
from nomenclator.nomenclature.tree import HSDocumentRef, HSSection


def test_build_research_context_groups_chapters_by_section(agent) -> None:
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
    agent,
    general_rules,
    chapter_mock,
    patch_candidate_chapters,
    patch_headings,
) -> None:
    """Chapter notes should be included regardless of retrieved headings."""

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

    agent._build_research_context = MagicMock(
        return_value=MagicMock(),
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

    note = MagicMock()
    note.to_dict.return_value = {
        "number": "1",
        "intro": "Chapter note",
    }

    chapter = chapter_mock(
        chapter_number=85,
        ref="8501-2022E",
        notes=[note],
    )

    agent._client.get_chapter.return_value = chapter

    heading = MagicMock(
        code="8507",
        description="Electric accumulators",
        subheadings=[],
    )

    agent._classification_analyst = MagicMock(
        return_value=HSClassificationSelectionOutputModel(
            candidates=[],
        ),
    )

    with (
        patch_candidate_chapters([MagicMock()]),
        patch_headings({85: [heading]}),
    ):
        agent.classify("lithium battery pack")

    _, context_arg = agent._classification_analyst.call_args.args

    assert context_arg.general_rules == [rule.to_dict() for rule in general_rules.rules]

    assert len(context_arg.chapters) == 1

    chapter = context_arg.chapters[0]

    assert chapter.chapter_number == 85
    assert chapter.title == "Electrical machinery and equipment"
    assert chapter.notes == [
        {
            "number": "1",
            "intro": "Chapter note",
        }
    ]

    assert len(chapter.headings) == 1

    heading = chapter.headings[0]
    assert heading.code == "8507"
    assert heading.description == "Electric accumulators"
    assert heading.subheadings == []


def test_classification_context_handles_chapter_without_headings(
    agent, general_rules, chapter_mock, patch_candidate_chapters
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

    chapter = chapter_mock(
        chapter_number=99,
        ref="9901-2022E",
        notes=[note],
    )

    agent._client.get_chapter.return_value = chapter

    agent._classification_analyst.return_value = HSClassificationSelectionOutputModel(
        candidates=[],
    )

    # No headings are retrieved for chapter 99.
    with (
        patch_candidate_chapters([MagicMock()]),
        patch.object(
            HSClassificationAgent,
            "_retrieve_headings",
            return_value={},
        ) as retrieve_headings,
    ):
        agent.classify("rare product")

    retrieve_headings.assert_called_once()

    (_, context_arg) = agent._classification_analyst.call_args.args

    assert context_arg.general_rules == [rule.to_dict() for rule in general_rules.rules]

    assert len(context_arg.chapters) == 1

    chapter = context_arg.chapters[0]

    assert chapter.chapter_number == 99
    assert chapter.title == "Electrical machinery and equipment"
    assert chapter.notes == [{"number": "1", "intro": "Note"}]
    assert chapter.headings == []
