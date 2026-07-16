import pytest

from nomenclator.retrieval.hybrid import RetrievalDocument, Retriever


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    """Create a retriever over a small synthetic corpus."""

    return Retriever(
        [
            RetrievalDocument(
                id="shirts",
                content="Men's knitted shirts",
                payload="shirts",
            ),
            RetrievalDocument(
                id="battery",
                content="Lithium ion battery",
                payload="battery",
            ),
            RetrievalDocument(
                id="banana",
                content="Fresh bananas",
                payload="banana",
            ),
        ],
        model_name="minishlab/potion-base-8M",
    )


def test_all_documents_are_preserved(
    retriever: Retriever,
) -> None:
    """All input documents should be indexed."""

    assert len(retriever._documents) == 3

    assert {document.id for document in retriever._documents} == {
        "shirts",
        "battery",
        "banana",
    }


def test_payloads_are_preserved(
    retriever: Retriever,
) -> None:
    """The retriever should preserve document payloads."""

    assert retriever._documents_by_id["battery"].payload == "battery"


def test_document_order_is_preserved(
    retriever: Retriever,
) -> None:
    """Documents should preserve their original order."""

    ids = [document.id for document in retriever._documents]

    assert ids == [
        "shirts",
        "battery",
        "banana",
    ]


def test_build_query_without_keywords() -> None:
    """Return the description unchanged when no keywords are provided."""
    assert (
        Retriever._build_query(
            "fresh bananas",
            None,
        )
        == "fresh bananas"
    )


def test_build_query_with_empty_keywords() -> None:
    """Return the description unchanged when no keywords are provided."""
    assert (
        Retriever._build_query(
            "fresh bananas",
            [],
        )
        == "fresh bananas"
    )


def test_build_query_with_keywords() -> None:
    """Return the description concatenated to keywords when they are provided."""
    assert (
        Retriever._build_query(
            "fresh bananas",
            ["fruit", "banana"],
        )
        == "fresh bananas fruit banana"
    )


def test_rrf_scores_empty(retriever: Retriever) -> None:
    assert retriever._rrf_scores({}) == {}


def test_rrf_scores_single(retriever: Retriever) -> None:
    scores = retriever._rrf_scores({"a": 0.9})

    assert list(scores) == ["a"]
    assert scores["a"] > 0


def test_rrf_scores_preserves_ranking(retriever: Retriever) -> None:
    scores = retriever._rrf_scores(
        {
            "a": 0.9,
            "b": 0.7,
            "c": 0.4,
        }
    )

    assert scores["a"] > scores["b"] > scores["c"]


def test_rrf_scores_depend_only_on_rank(
    retriever: Retriever,
) -> None:
    """RRF should depend only on document rank, not score magnitude."""
    scores1 = retriever._rrf_scores(
        {
            "a": 1000,
            "b": 2,
            "c": 1,
        }
    )

    scores2 = retriever._rrf_scores(
        {
            "a": 0.9,
            "b": 0.8,
            "c": 0.7,
        }
    )

    assert scores1 == scores2


def test_search_respects_limit(
    retriever: Retriever,
) -> None:
    """Search should return at most the requested number of documents."""

    results = retriever.search(
        "shirts",
        limit=2,
    )

    assert len(results) <= 2


@pytest.fixture(scope="module")
def integration_retriever() -> Retriever:
    """Create a retriever over a small synthetic corpus."""

    documents = [
        RetrievalDocument(
            id="shirts",
            content="Men's knitted cotton shirts and other apparel.",
            payload={"category": "apparel"},
        ),
        RetrievalDocument(
            id="batteries",
            content="Lithium-ion battery packs for electric vehicles.",
            payload={"category": "battery"},
        ),
        RetrievalDocument(
            id="fruit",
            content="Fresh bananas and tropical fruit.",
            payload={"category": "fruit"},
        ),
    ]

    return Retriever(
        documents,
        model_name="minishlab/potion-base-8M",
    )


@pytest.mark.parametrize(
    ("description", "keywords", "expected_id", "expected_category"),
    [
        (
            "men's cotton knitted shirts",
            ["cotton", "knitted"],
            "shirts",
            "apparel",
        ),
        (
            "electric vehicle lithium ion battery pack",
            ["battery", "electric vehicle"],
            "batteries",
            "battery",
        ),
        (
            "fresh bananas",
            ["banana"],
            "fruit",
            "fruit",
        ),
    ],
)
def test_hybrid_search(
    integration_retriever: Retriever,
    description: str,
    keywords: list[str],
    expected_id: str,
    expected_category: str,
) -> None:
    """Return the most relevant document for representative queries."""

    results = integration_retriever.search(
        description,
        keywords=keywords,
        limit=1,
    )

    assert results

    result = results[0]

    assert result.document.id == expected_id
    assert result.document.payload["category"] == expected_category
