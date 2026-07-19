"""Shared pytest configuration for the nomenclator test suite."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nomenclator.agent import HSClassificationAgent
from nomenclator.nomenclature.client import NomenclatureClient
from nomenclator.nomenclature.rules import HSGeneralRule, HSGeneralRules
from nomenclator.nomenclature.tree import HSDocumentRef


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: tests that call the live WCO nomenclature source",
    )


@pytest.fixture
def nomenclature_client() -> NomenclatureClient:  # type: ignore
    """Provide a live nomenclature client."""
    with NomenclatureClient() as client:
        yield client


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


@pytest.fixture
def heading_mock():
    """Factory for mocked HSHeading objects."""

    def factory(heading_dict: dict) -> MagicMock:
        heading = MagicMock()
        heading.to_dict.return_value = heading_dict
        return heading

    return factory


@pytest.fixture
def chapter_mock():
    """Factory for mocked HSChapter objects."""

    def factory(
        *,
        chapter_number: int = 85,
        title: str = "Electrical machinery and equipment",
        ref: str = "8501-2022E",
        notes: list | None = None,
    ) -> MagicMock:
        chapter = MagicMock()
        chapter.chapter_number = chapter_number
        chapter.title = title
        chapter.document.ref = ref
        chapter.notes = notes if notes is not None else []
        chapter.headings = []
        return chapter

    return factory


@pytest.fixture
def patch_candidate_chapters():
    """Factory for patching _retrieve_chapters()."""

    def factory(retrieved: list):
        return patch.object(
            HSClassificationAgent,
            "_retrieve_chapters",
            return_value=retrieved,
        )

    return factory


@pytest.fixture
def patch_headings():
    """Factory for patching _retrieve_headings()."""

    def factory(headings_by_chapter: dict[int, list[MagicMock]]):
        return patch.object(
            HSClassificationAgent,
            "_retrieve_headings",
            return_value=headings_by_chapter,
        )

    return factory


@pytest.fixture
def search_result(heading_mock):
    """Factory for mocked SearchResult objects."""

    def factory(
        chunk_id: str,
        chapter_number: int,
        heading_dict: dict,
    ) -> MagicMock:
        heading = heading_mock(heading_dict)

        document = MagicMock()
        document.id = chunk_id
        document.payload = heading

        result = MagicMock()
        result.document = document

        return result

    return factory
