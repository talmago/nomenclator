"""Generic hybrid retrieval engine.

This module provides hybrid retrieval over arbitrary collections of documents
by combining semantic similarity with BM25 keyword search.

The retriever indexes an iterable of ``RetrievalDocument`` objects and is
agnostic to the underlying domain. It can be used for HS nomenclature,
documentation, legal texts, product catalogs, or any other textual corpus.

The retrieval pipeline is:

    Query
      |
      v
Description + keywords
      |
      v
HybridRetriever
      |
      +----------------+
      |                |
      v                v
Semantic retrieval   BM25 retrieval
 (embeddings)        (keyword search)
      |                |
      +-------+--------+
              |
              v
      Reciprocal Rank Fusion
              |
              v
     Ranked search results

Semantic retrieval identifies conceptually similar documents, while BM25
preserves strong lexical matches for important terminology. The two rankings
are combined using Reciprocal Rank Fusion (RRF) to produce robust retrieval
results across a wide range of query types.

The retriever is independent of any particular document model or application.
It operates solely on ``RetrievalDocument`` instances and performs no network
access, document parsing, or application-specific reasoning.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import re

import bm25s
from model2vec import StaticModel
import numpy as np
import numpy.typing as npt

from nomenclator.models.search import RetrievalDocument, SearchResult, T
from nomenclator.models.tree import HSDocumentRef
from nomenclator.retrieval.dense import SelectableBasicBackend, load_model


class Retriever:
    """Hybrid semantic and lexical document retriever.

    The retriever builds a searchable index over an iterable of
    ``RetrievalDocument`` instances. It is domain-agnostic and performs no network
    access, document parsing, or application-specific reasoning.

    Search combines two complementary retrieval strategies:

    * Semantic retrieval uses sentence embeddings to identify conceptually similar
    documents.
    * Keyword retrieval uses BM25 to preserve exact terminology matches.

    The two rankings are combined using Reciprocal Rank Fusion (RRF) to produce a
    single ranked result list.

    Args:
        documents: Documents to index for retrieval.
    """

    _RRF_K = 60

    def __init__(
        self,
        documents: Iterable[RetrievalDocument[T]],
        *,
        model: StaticModel | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            documents: Searchable documents to index.
            model: Pre-loaded embedding model.
            model_name: Embedding model identifier to load when ``model`` is not
                provided.

        Raises:
            ValueError: If both ``model`` and ``model_name`` are provided.
        """

        if model is not None and model_name is not None:
            raise ValueError("Provide either model or model_name, not both")

        if model is None:
            model, _ = load_model(model_name)

        self._model = model
        self._documents = list(documents)
        self._documents_by_id = {document.id: document for document in self._documents}
        self._bm25_index = self._build_bm25_index()
        self._semantic_index = self._build_semantic_index(model)

    def _build_bm25_index(self) -> bm25s.BM25:
        """Build the BM25 lexical retrieval index.

        The index is built from the searchable text representation of each HS
        retrieval document. The same tokenization strategy used during query
        retrieval must be applied here to keep document and query tokens aligned.

        Returns:
            A configured BM25 retrieval index.
        """

        corpus_tokens = [
            self._tokenize(document.content) for document in self._documents
        ]

        index = bm25s.BM25()

        index.index(corpus_tokens)

        return index

    def _build_semantic_index(
        self,
        model: StaticModel,
    ) -> SelectableBasicBackend:
        """Build the semantic vector retrieval index.

        The HS retrieval documents are embedded using the provided static
        embedding model and stored in a cosine similarity backend for semantic
        search.

        Args:
            model: Static embedding model used to encode retrieval documents.

        Returns:
            A cosine similarity vector index containing document embeddings.
        """

        embeddings = np.array(
            model.encode(
                [document.content for document in self._documents],
                use_multiprocessing=False,
            ),
            dtype=np.float32,
        )

        return SelectableBasicBackend(embeddings, {})

    def search(
        self,
        description: str,
        *,
        keywords: Sequence[str] | None = None,
        limit: int = 10,
    ) -> list[SearchResult[T]]:
        """Search the indexed documents.

        Args:
            description: Primary search query.
            keywords: Optional additional search keywords.
            limit: Maximum number of results to return.

        Returns:
            Ranked search results.
        """

        query = self._build_query(
            description,
            keywords,
        )

        candidate_count = limit * 5

        semantic_scores = self._search_semantic(
            query,
            top_k=candidate_count,
        )

        bm25_scores = self._search_bm25(
            query,
            top_k=candidate_count,
        )

        normalized_semantic = self._rrf_scores(
            semantic_scores,
        )

        normalized_keyword = self._rrf_scores(
            bm25_scores,
        )

        all_indexes = set(normalized_semantic) | set(normalized_keyword)

        combined_scores = {
            index: (
                0.5 * normalized_semantic.get(index, 0.0)
                + 0.5 * normalized_keyword.get(index, 0.0)
            )
            for index in all_indexes
        }

        ranked = sorted(
            combined_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]

        results: list[SearchResult[T]] = []

        for document_id, score in ranked:
            document = self._documents_by_id[document_id]

            results.append(
                SearchResult(
                    document=document,
                    score=score,
                    semantic_score=semantic_scores.get(document_id),
                    bm25_score=bm25_scores.get(document_id),
                )
            )

        return results

    def _search_semantic(
        self,
        query: str,
        *,
        top_k: int,
    ) -> dict[str, float]:
        """Run semantic retrieval over the HS document index.

        The query is embedded using the configured static embedding model and
        compared against the pre-built vector index. The underlying vector backend
        returns cosine distances, which are converted into cosine similarity scores
        where higher values indicate greater semantic similarity.

        Args:
            query: Retrieval query text.
            top_k: Maximum number of semantic candidates to retrieve.

        Returns:
            Mapping of document IDs to cosine similarity scores.
        """

        query_embedding = self._model.encode([query])

        results = self._semantic_index.query(
            query_embedding,
            k=top_k,
        )[0]

        indices, distances = results

        return {
            self._documents[index].id: 1.0 - float(distance)
            for index, distance in zip(indices, distances)
        }

    def _search_bm25(
        self,
        query: str,
        *,
        top_k: int,
    ) -> dict[str, float]:
        """Run lexical BM25 retrieval.

        The query is tokenized using a lightweight whitespace-based tokenizer
        before being evaluated against the BM25 index.

        Args:
            query: Combined product description and keyword query.
            top_k: Maximum number of lexical candidates to return.

        Returns:
            Mapping of document IDs to BM25 scores.
        """
        tokens = [token.lower() for token in re.findall(r"\w+", query) if token.strip()]

        if not tokens:
            return {}

        scores = self._bm25_index.get_scores(tokens)

        indices = self._sort_top_k(
            scores,
            top_k,
        )

        return {
            self._documents[index].id: float(scores[index])
            for index in indices
            if scores[index] > 0
        }

    @staticmethod
    def _sort_top_k(
        scores: npt.NDArray,
        top_k: int,
    ) -> npt.NDArray[np.int_]:
        """Return indices of the top-k highest scoring documents.

        Uses partial sorting for efficiency when only a small subset of results
        is required from a larger index.

        Args:
            scores: Array of retrieval scores.
            top_k: Maximum number of indices to return.

        Returns:
            Array containing document indices ordered by descending score.
        """

        if top_k >= len(scores):
            return np.argsort(-scores)

        partitioned = np.argpartition(
            -scores,
            kth=top_k - 1,
        )[:top_k]

        return partitioned[np.argsort(-scores[partitioned])]

    def _rrf_scores(
        self,
        scores: dict[str, float],
    ) -> dict[str, float]:
        """Convert raw retrieval scores into Reciprocal Rank Fusion scores.

        RRF removes the dependency on the absolute scale of the underlying
        retrieval systems by converting ranked results into reciprocal rank
        contributions.

        The highest-scoring document receives rank 1 and therefore the largest
        RRF score. Documents that do not appear in the input scores receive no
        contribution.

        Args:
            scores: Mapping of document identifiers to raw retrieval scores.

        Returns:
            Mapping of document identifiers to RRF scores.
        """

        if not scores:
            return {}

        ranked = sorted(
            scores,
            key=lambda key: -scores[key],
        )

        return {
            document_id: 1.0 / (self._RRF_K + rank)
            for rank, document_id in enumerate(ranked, 1)
        }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25 retrieval.

        Args:
            text: Input text.

        Returns:
            Normalized token list.
        """

        return [token.lower() for token in re.findall(r"\w+", text) if token.strip()]

    @staticmethod
    def _chapter_text(
        *,
        section_label: str,
        section_title: str,
        chapter: HSDocumentRef,
    ) -> str:
        """Build the searchable text representation of an HS chapter.

        Includes HS hierarchy context so semantic retrieval can distinguish
        between broad categories and specific chapter concepts.
        """

        return " ".join(
            [
                f"Section: {section_label}",
                f"Section description: {section_title}",
                f"Chapter: {chapter.title}",
            ]
        )

    @staticmethod
    def _build_query(
        description: str,
        keywords: list[str] | None,
    ) -> str:
        """Build the retrieval query text.

        The query combines the main product description with optional keywords.
        This allows semantic retrieval to capture concepts while keyword search
        benefits from explicit terminology.

        Args:
            description: Main product description.
            keywords: Additional retrieval terms.

        Returns:
            Normalized retrieval query.
        """

        parts = [description]

        if keywords:
            parts.extend(keywords)

        return " ".join(part.strip() for part in parts if part and part.strip())
