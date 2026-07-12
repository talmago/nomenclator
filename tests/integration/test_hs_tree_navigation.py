"""Integration tests for HS navigation retrieval."""

import pytest

from nomenclator.models.search import RetrievalDocument
from nomenclator.nomenclature.client import NomenclatureClient
from nomenclator.retrieval import Retriever


@pytest.fixture(scope="module")
def hs_retriever() -> Retriever:
    """Create a retriever over HS navigation documents."""

    client = NomenclatureClient()

    tree = client.get_tree()

    documents = [
        RetrievalDocument(
            id=chapter.ref,
            content=(f"{section.label} {section.title} {chapter.title}"),
            payload=chapter,
        )
        for section in tree.sections
        for chapter in section.chapters
        if chapter.ref
    ]

    return Retriever(
        documents,
        model_name="minishlab/potion-base-8M",
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("description", "keywords", "expected_refs"),
    [
        (
            "men's cotton knitted shirts",
            [
                "cotton",
                "knitted",
                "shirts",
                "garments",
            ],
            {
                "1161-2022E",
                "1162-2022E",
            },
        ),
        (
            "electric vehicle lithium ion battery pack",
            [
                "battery",
                "lithium",
                "electric vehicle",
            ],
            {
                "1787-2022E",
            },
        ),
        (
            "fresh bananas",
            [
                "fruit",
                "banana",
            ],
            {
                "0208-2022E",
            },
        ),
    ],
)
def test_hs_navigation_retrieval(
    hs_retriever: Retriever,
    description: str,
    keywords: list[str],
    expected_refs: set[str],
) -> None:
    """Retrieve relevant HS chapters from the official nomenclature."""

    results = hs_retriever.search(
        description,
        keywords=keywords,
        limit=5,
    )

    assert results

    result_refs = {result.document.id for result in results}

    assert result_refs & expected_refs, (
        f"Expected one of {expected_refs}, got {result_refs}"
    )
