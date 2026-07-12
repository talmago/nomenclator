from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class RetrievalDocument[T]:
    id: str
    content: str
    payload: T


@dataclass(slots=True)
class SearchResult[T]:
    """Result returned by the hybrid retriever."""

    document: RetrievalDocument[T]
    score: float
    semantic_score: float | None = None
    bm25_score: float | None = None
