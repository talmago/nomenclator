from unittest.mock import MagicMock, patch

from nomenclator.agent import HSClassificationAgent
from nomenclator.nomenclature.chapter import HSChapter
from nomenclator.nomenclature.tree import HSDocumentRef, HSHeading, HSSubheading


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


def test_heading_chunks_produces_heading_chunks() -> None:
    """Each heading should become one retrieval chunk with heading as payload."""

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


def test_select_headings_guarantees_one_per_chapter(agent, search_result) -> None:
    """Every chapter with chunks must get at least one heading (floor)."""

    # Global ranking puts all of chapter 90 first; chapter 85's only chunk
    # is ranked last. The floor must still give chapter 85 one heading.
    results = [
        search_result("90:9001", 90, {"code": "9001"}),
        search_result("90:9002", 90, {"code": "9002"}),
        search_result("90:9003", 90, {"code": "9003"}),
        search_result("85:8507", 85, {"code": "8507"}),
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
    agent,
    search_result,
) -> None:
    """Total selected headings must never exceed the global budget."""

    results = [
        search_result(f"85:850{i}", 85, {"code": f"850{i}"}) for i in range(1, 6)
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
    agent,
    search_result,
) -> None:
    """When the budget is smaller than the number of chapters, the floor phase
    stops at the budget, so only the highest-ranked chapters are represented."""

    results = [
        search_result("90:9001", 90, {"code": "9001"}),
        search_result("85:8507", 85, {"code": "8507"}),
        search_result("84:8401", 84, {"code": "8401"}),
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
