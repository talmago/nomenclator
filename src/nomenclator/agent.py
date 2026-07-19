"""DSPy-based HS classification pipeline.

The pipeline combines deterministic HS document retrieval with LLM-based legal
reasoning to produce ranked HS classification candidates.

Architecture:

```
Raw product description
        |
        v
+-----------------+
| Product Analyst |
+-----------------+
        |
        v
  Product facts
        |
        v
+--------------------------------+
| HS Navigation Document Builder  |
+--------------------------------+
        |
        v
RetrievalDocument[HSDocumentRef][]
        |
        v
+------------------+
| Hybrid Retriever |
+------------------+
        |
        v
SearchResult[HSDocumentRef][]
        |
        v
+------------------------+
| Research Context Builder|
+------------------------+
        |
        v
  HSResearchContext
        |
        v
+------------------+
| Research Analyst |
+------------------+
        |
        v
ResearchCandidateModel[]
        |
        v
+-------------------------+
| Classification Analyst  |
+-------------------------+
        |
        v
HSCodeCandidateModel[]
```

The Product Analyst extracts structured product facts from the raw product
description.

The HS Navigation Document Builder converts the parsed HS nomenclature tree
into searchable documents. The Hybrid Retriever performs deterministic
semantic and lexical retrieval to identify potentially relevant HS chapters.

The Research Context Builder groups retrieved chapters by HS section and
enriches them with the corresponding section notes, providing legal context
for pathway selection without loading full chapter documents.

The Research Analyst evaluates the retrieved context, filters irrelevant
chapters, and identifies the most plausible HS classification pathways.

The Classification Analyst loads only the shortlisted chapters, splits each
chapter into heading-level chunks, and indexes all chunks from all
shortlisted chapters together in a single hybrid retriever. Only the globally
most relevant chunks are kept, bounded by a global chunk budget
(``max_chunks``), with each shortlisted chapter guaranteed at least one
heading chunk. Chapter notes are always included in full. The analyst then
performs detailed legal classification over this compacted context to
produce ranked HS code candidates. Capping the total number of chunks keeps
the Classification Analyst prompt compact even for very long or dense
chapters and regardless of how many chapters were shortlisted.

The retrieval layer is deterministic and independent from LLM reasoning,
allowing retrieval quality and classification reasoning to be evaluated
separately.
"""

from __future__ import annotations

from dataclasses import dataclass

from nomenclator.exceptions import (
    HSClassificationAnalysisError,
    HSClassificationPipelineError,
    HSInitializationError,
    HSNoCandidatesFoundError,
    HSProductAnalysisError,
    HSResearchAnalysisError,
)
from nomenclator.models.classification import (
    HSClassificationOutputModel,
    HSCodeCandidateModel,
)
from nomenclator.models.navigation import (
    HSResearchCandidateModel,
    HSResearchChapterContext,
    HSResearchContext,
    HSResearchOutputModel,
    HSResearchSectionContext,
)
from nomenclator.models.product_facts import ProductFactsModel
from nomenclator.nomenclature.chapter import HSChapter
from nomenclator.nomenclature.client import NomenclatureClient
from nomenclator.nomenclature.tree import HSDocumentRef, HSHeading
from nomenclator.retrieval.hybrid import RetrievalDocument, Retriever, SearchResult
from nomenclator.tasks.classification_analyst import ClassificationAnalyst
from nomenclator.tasks.product_analyst import ProductAnalyst
from nomenclator.tasks.research_analyst import ResearchAnalyst
from nomenclator.usage import ensure_dspy_lm


@dataclass(slots=True)
class HSClassificationResult:
    """End-to-end result of the HS classification pipeline.

    Wraps the final classification together with the intermediate artifacts
    produced by each upstream stage, so callers can inspect the navigation
    and reasoning that led to the candidates instead of only seeing the
    final codes.

    Attributes:
        facts: Structured product facts extracted by the Product Analyst.
        keywords: Retrieval keywords derived from the product facts.
        retrieved: Raw chapter retrieval results from the Nomenclature
            Retriever (semantic + BM25 hybrid search).
        navigation: Ranked chapter candidates produced by the Research
            Analyst.
        classification: Final ranked HS code candidates produced by the
            Classification Analyst.
    """

    facts: ProductFactsModel
    keywords: list[str]
    retrieved: list[SearchResult[HSDocumentRef]]
    navigation: HSResearchOutputModel
    classification: HSClassificationOutputModel

    @property
    def candidates(self) -> list[HSCodeCandidateModel]:
        """Ranked HS code candidates.

        Convenience alias for ``classification.candidates`` so callers can
        keep using ``result.candidates`` directly.
        """

        return self.classification.candidates


class HSClassificationAgent:
    """DSPy-based HS classification pipeline.

    The pipeline consists of:

        Raw product description
                |
                v
        Product Analyst
                |
                v
        Product facts
                |
                v
        HS navigation document retrieval
                |
                v
        Hybrid Retriever
                |
                v
        Candidate HS chapters
                |
                v
        Research Analyst
                |
                v
        Ranked chapter candidates
                |
                v
        Classification context builder
        (global heading-chunk budget)
                |
                v
        Classification Analyst
                |
                v
        HS code candidates

    The navigation retrieval layer is deterministic and uses semantic and lexical
    search over HS nomenclature documents. It is independent from LLM reasoning.

    After the Research Analyst shortlists chapters, the classification context
    builder splits each shortlisted chapter into heading-level chunks and indexes
    them together in a single hybrid retriever. Only the globally most relevant
    chunks are kept, bounded by ``max_chunks``, with each shortlisted chapter
    guaranteed at least one heading chunk; chapter notes are always included in
    full. This keeps the Classification Analyst prompt compact regardless of how
    many chapters were shortlisted or how dense they are.

    DSPy modules are responsible for analyzing retrieved context, applying HS
    classification logic, and producing ranked classification candidates.
    """

    def __init__(
        self,
        *,
        client: NomenclatureClient | None = None,
        model_name: str = "minishlab/potion-base-8M",
        retrieval_limit: int = 5,
        max_candidates: int = 3,
        max_chunks: int = 10,
    ) -> None:
        """Initialize the classification agent.

        Args:
            client: HS nomenclature client.

            model_name: Embedding model used by the retrievers.

            retrieval_limit: Maximum number of HS chapters retrieved by the hybrid
                retriever and passed to the research analyst.

            max_candidates: Maximum number of candidates returned at each
                ranking stage. It caps both the HS chapter pathways shortlisted
                by the research analyst and the ranked HS subheading candidates
                returned by the classification analyst. A larger value surfaces
                more competing classification pathways but increases prompt
                size and cost.

            max_chunks: Global budget for the total number of heading chunks
                loaded into the Classification Analyst prompt across all
                shortlisted chapters. All shortlisted chapters are split into
                heading-level chunks and indexed together by a single retriever;
                only the ``max_chunks`` most relevant chunks overall are kept
                instead of loading the whole chapters. Each shortlisted chapter
                is guaranteed at least one heading chunk (when it has any), so
                the Research Analyst's shortlisting is respected. This trades
                classification accuracy against model cost: a smaller value
                yields a more compact prompt and lower token cost but risks
                dropping the heading that contains the correct 6-digit
                subheading, while a larger value improves recall of the
                relevant heading at the expense of a larger prompt. Because the
                budget is global, it bounds the total context size regardless of
                how many chapters were shortlisted or how dense they are.
                Defaults to ``10``.
        """
        if retrieval_limit <= 0:
            raise ValueError("retrieval_limit must be greater than zero")

        if max_candidates <= 0:
            raise ValueError("max_candidates must be greater than zero")

        if max_chunks <= 0:
            raise ValueError("max_chunks must be greater than zero")

        self._client = client or NomenclatureClient()
        self._retrieval_limit = retrieval_limit
        self._model_name = model_name
        self._max_candidates = max_candidates
        self._max_chunks = max_chunks

        try:
            self._tree = self._client.get_tree()
        except Exception as exc:
            raise HSInitializationError(
                "Failed to initialize HS classification resources"
            ) from exc

        self._product_analyst = ProductAnalyst()
        self._research_analyst = ResearchAnalyst(max_candidates=max_candidates)
        self._classification_analyst = ClassificationAnalyst()

    def classify(
        self,
        description: str,
        *,
        user_hs_codes: list[str] | None = None,
    ) -> HSClassificationResult:
        """Classify a product description.

        Args:
            description: Raw product description.
            user_hs_codes: Optional user-provided HS hints.

        Returns:
            End-to-end classification result, including the final ranked HS
            code candidates (``.candidates``) and the intermediate artifacts
            from each upstream stage (facts, keywords, retrieved chapters,
            and navigation).

        Raises:
            HSInitializationError: If no DSPy language model is configured.
        """

        ensure_dspy_lm()

        # --------------------------------------------------
        # Step 1: Product Analyst
        # --------------------------------------------------

        try:
            facts = self._product_analyst(
                description,
                hints=user_hs_codes or [],
            )
        except Exception as exc:
            raise HSProductAnalysisError("Failed to extract product facts") from exc

        # --------------------------------------------------
        # Step 2: Retrieval
        # --------------------------------------------------

        keywords = self._retrieval_keywords(
            facts,
            user_hs_codes,
        )

        retrieved = self._retrieve_chapters(
            facts.normalized_description,
            keywords=keywords,
        )

        if not retrieved:
            raise HSNoCandidatesFoundError(
                "No HS chapter candidates found for product description"
            )

        # --------------------------------------------------
        # Step 3: Research Analyst
        # --------------------------------------------------

        research_context = self._build_research_context(
            retrieved,
        )

        try:
            navigation = self._research_analyst(
                facts,
                research_context,
            )
        except Exception as exc:
            raise HSResearchAnalysisError(
                "Failed to analyze retrieved HS candidates"
            ) from exc

        # --------------------------------------------------
        # Step 4: Classification Analyst
        # --------------------------------------------------

        try:
            chapter_context = self._build_classification_context(
                navigation.candidates,
                facts=facts,
                keywords=keywords,
            )
            general_rules = [
                rule.to_dict() for rule in self._client.get_general_rules().rules
            ]
        except Exception as exc:
            raise HSClassificationPipelineError(
                "Failed to load HS classification context"
            ) from exc

        try:
            classification = self._classification_analyst(
                facts,
                chapter_context,
                general_rules,
                self._max_candidates,
            )
        except Exception as exc:
            raise HSClassificationAnalysisError(
                "Failed to produce HS classification candidates"
            ) from exc

        return HSClassificationResult(
            facts=facts,
            keywords=keywords,
            retrieved=retrieved,
            navigation=navigation,
            classification=classification,
        )

    def _build_research_context(
        self,
        retrieved: list[SearchResult[HSDocumentRef]],
    ) -> HSResearchContext:
        """Build legal context for the Research Analyst.

        The returned context groups retrieved chapter candidates by HS section and
        includes the corresponding section notes. This provides the Research
        Analyst with sufficient legal context to identify plausible classification
        pathways while avoiding the cost of loading full chapter documents.

        Args:
            retrieved: Chapter candidates returned by the hybrid retriever.

        Returns:
            Structured research context containing candidate chapters grouped by
            section together with their section notes.
        """

        sections: dict[str, HSResearchSectionContext] = {}

        for result in retrieved:
            document = result.document.payload

            if document.ref is None:
                continue

            section = self._client.get_section_for_chapter(
                document.ref,
            )

            if section.id not in sections:
                try:
                    section_notes = self._client.get_section_notes(section.id).to_dict()
                except ValueError:
                    section_notes = None

                sections[section.id] = HSResearchSectionContext(
                    section_id=section.id,
                    section_title=section.title,
                    section_notes=section_notes,
                    chapters=[],
                )

            sections[section.id].chapters.append(
                HSResearchChapterContext(
                    chapter_ref=document.ref,
                    chapter_title=document.title,
                    retrieval_score=result.score,
                )
            )

        return HSResearchContext(
            sections=list(sections.values()),
        )

    def _build_classification_context(
        self,
        candidates: list[HSResearchCandidateModel],
        *,
        facts: ProductFactsModel,
        keywords: list[str],
    ) -> list[dict]:
        """Build compact classification context for all shortlisted chapters.

        All shortlisted chapters are split into heading-level chunks and indexed
        together by a single hybrid retriever. Only the globally most relevant
        ``max_chunks`` chunks are kept across all chapters, which bounds the
        total Classification Analyst prompt size regardless of how many chapters
        were shortlisted or how dense they are.

        Each shortlisted chapter is guaranteed at least one heading chunk (when
        it has any headings) so the Research Analyst's shortlisting is
        respected.

        Chapter notes are always included for every shortlisted chapter,
        regardless of the retrieved chunks, because they apply to the whole
        chapter and are required for legal reasoning.

        Args:
            candidates: Ranked chapter candidates from the Research Analyst.
            facts: Structured product facts used as the retrieval query.
            keywords: Retrieval keywords combined with the product description.

        Returns:
            One classification context dict per candidate, each containing
            chapter metadata, all chapter notes, and the retrieved heading
            hierarchy for that chapter.
        """

        chapters = [
            self._client.get_chapter(candidate.chapter_ref) for candidate in candidates
        ]

        headings_by_chapter = self._retrieve_headings(
            chapters,
            description=facts.normalized_description,
            keywords=keywords,
        )

        return [
            {
                "chapter_number": chapter.chapter_number,
                "title": chapter.title,
                "notes": [note.to_dict() for note in chapter.notes],
                "headings": [
                    heading.to_dict()
                    for heading in headings_by_chapter.get(chapter.chapter_number, [])
                ],
            }
            for chapter in chapters
        ]

    def _retrieve_chapters(
        self,
        description: str,
        *,
        keywords: list[str],
    ) -> list[SearchResult[HSDocumentRef]]:
        """Retrieve candidate HS chapters for a product description.

        Builds the HS navigation retriever inline over the parsed nomenclature
        tree and searches it with the product description and retrieval
        keywords. The retriever is constructed on demand rather than held as
        agent state, since it is only used at this stage of the pipeline.

        Args:
            description: Normalized product description used as the query.
            keywords: Lexical retrieval keywords combined with the query.

        Returns:
            Ranked retrieval results whose payloads are HS chapter references.
        """

        documents = [
            RetrievalDocument(
                id=chapter.ref,
                content=self._chapter_content(
                    section_label=section.label,
                    section_title=section.title,
                    chapter=chapter,
                ),
                payload=chapter,
            )
            for section in self._tree.sections
            for chapter in section.chapters
            if chapter.ref
        ]

        retriever = Retriever(
            documents,
            model_name=self._model_name,
        )

        return retriever.search(
            description,
            keywords=keywords,
            limit=self._retrieval_limit,
        )

    def _retrieve_headings(
        self,
        chapters: list[HSChapter],
        *,
        description: str,
        keywords: list[str],
    ) -> dict[int, list[HSHeading]]:
        """Retrieve the most relevant headings across all shortlisted chapters.

        A single hybrid retriever is built over the heading chunks of every
        shortlisted chapter and searched once with the product facts. The
        retriever returns a global ranking over all chunks. Selection then
        guarantees one chunk per chapter that has headings (the floor), and
        fills the remaining budget with the next most relevant chunks overall.

        Args:
            chapters: Parsed shortlisted chapters to search within.
            description: Normalized product description used as the query.
            keywords: Retrieval keywords combined with the product description.

        Returns:
            Mapping of chapter number to the selected headings for that
            chapter. Chapters without headings are absent from the mapping.
        """

        all_chunks: list[RetrievalDocument[HSHeading]] = []
        chunk_to_chapter: dict[str, int] = {}
        chapters_with_chunks: set[int] = set()

        for chapter in chapters:
            chunks = self._split_chapter_into_chunks(chapter)
            if not chunks:
                continue
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_to_chapter[chunk.id] = chapter.chapter_number
            chapters_with_chunks.add(chapter.chapter_number)

        if not all_chunks:
            return {}

        retriever = Retriever(
            all_chunks,
            model_name=self._model_name,
        )

        results: list[SearchResult[HSHeading]] = retriever.search(
            description,
            keywords=keywords,
            limit=len(all_chunks),
        )

        return self._select_headings(
            results,
            chunk_to_chapter=chunk_to_chapter,
            chapters_with_chunks=chapters_with_chunks,
            budget=self._max_chunks,
        )

    @staticmethod
    def _select_headings(
        results: list[SearchResult[HSHeading]],
        *,
        chunk_to_chapter: dict[str, int],
        chapters_with_chunks: set[int],
        budget: int,
    ) -> dict[int, list[HSHeading]]:
        """Select up to ``budget`` headings with a one-per-chapter floor.

        Selection is performed in two phases over the globally ranked results:

        1. Floor: take the highest-ranked chunk of each chapter so every
           chapter that has headings is represented by at least one heading.
           This phase stops early once the budget is reached.
        2. Fill: take the next highest-ranked chunks overall, skipping chunks
           already selected, until the budget is reached or results are
           exhausted.

        Args:
            results: Globally ranked retrieval results.
            chunk_to_chapter: Mapping of chunk id to chapter number.
            chapters_with_chunks: Chapter numbers that contributed chunks.
            budget: Maximum total number of headings to select.

        Returns:
            Mapping of chapter number to selected headings, in selection
            order.
        """

        selected_by_chapter: dict[int, list[HSHeading]] = {
            chapter_number: [] for chapter_number in chapters_with_chunks
        }
        selected_ids: set[str] = set()
        represented: set[int] = set()

        # Phase 1: guarantee one heading per chapter that has headings.
        for result in results:
            if len(selected_ids) >= budget:
                break
            document = result.document
            chapter_number = chunk_to_chapter[document.id]
            if chapter_number in represented:
                continue
            selected_by_chapter[chapter_number].append(document.payload)
            selected_ids.add(document.id)
            represented.add(chapter_number)
            if len(represented) == len(chapters_with_chunks):
                break

        # Phase 2: fill the remaining budget with the next best chunks.
        for result in results:
            if len(selected_ids) >= budget:
                break
            document = result.document
            if document.id in selected_ids:
                continue
            chapter_number = chunk_to_chapter[document.id]
            selected_by_chapter[chapter_number].append(document.payload)
            selected_ids.add(document.id)

        return selected_by_chapter

    @staticmethod
    def _split_chapter_into_chunks(
        chapter: HSChapter,
    ) -> list[RetrievalDocument[HSHeading]]:
        """Split a chapter into heading-level retrieval chunks.

        Each heading together with its subheadings becomes one retrieval
        document. This allows the Classification Analyst to retrieve only the
        relevant headings of a chapter instead of loading the entire chapter
        into context, which keeps prompts compact and prevents the context from
        blowing up for long chapters.

        Chapter notes are intentionally excluded from the chunks; they apply to
        the whole chapter and are added to the classification context separately
        by :meth:`_build_classification_context`.

        Args:
            chapter: Parsed chapter to chunk.

        Returns:
            Retrieval documents, one per heading, with the heading as payload.
        """

        return [
            RetrievalDocument(
                id=f"{chapter.chapter_number}:{heading.code}",
                content=HSClassificationAgent._heading_chunk_content(heading),
                payload=heading,
            )
            for heading in chapter.headings
        ]

    @staticmethod
    def _chapter_content(
        *,
        section_label: str,
        section_title: str,
        chapter: HSDocumentRef,
    ) -> str:
        """Build the indexed content for an HS chapter.

        The content is designed for semantic and lexical retrieval rather than
        presentation. It combines the section and chapter titles into a concise,
        natural-language document.

        Args:
            section_label: HS section label (for example, ``"SECTION XI"``).
            section_title: HS section title.
            chapter: Chapter reference.

        Returns:
            Indexed content for the retrieval document.
        """

        return (
            f"{section_label}\n{section_title}"
            f"\n\nChapter {chapter.ref}\n{chapter.title}"
        )

    @staticmethod
    def _heading_chunk_content(heading: HSHeading) -> str:
        """Build the indexed content for a heading chunk.

        The content combines the heading code and description with the codes
        and descriptions of its subheadings. Intermediate grouping labels
        carried by subheadings are included when present so that lexical and
        semantic retrieval can match against grouping terminology.

        Args:
            heading: Heading to render.

        Returns:
            Searchable text representation of the heading chunk.
        """

        parts = [f"Heading {heading.code}", heading.description]

        for subheading in heading.subheadings:
            label = " ".join(subheading.group_path).strip()
            description = subheading.description
            if label:
                description = f"{label}: {description}" if description else label
            parts.append(f"{subheading.code} {description}".strip())

        return "\n".join(part for part in parts if part)

    @staticmethod
    def _retrieval_keywords(
        facts: ProductFactsModel,
        user_hs_codes: list[str] | None,
    ) -> list[str]:
        """Build retrieval keywords from extracted product facts.

        Creates a compact set of lexical retrieval terms for the hybrid retriever.
        The keywords combine Product Analyst output with selected structured
        attributes that are likely to improve HS nomenclature matching.

        Included attributes are intentionally limited to high-signal fields:
        product category, product type, and material. Additional classification
        hints such as part/component status and user-provided HS codes are included
        when available.

        Duplicate and empty values are removed while preserving insertion order.

        Args:
            facts: Structured product facts extracted by the Product Analyst.
            user_hs_codes: Optional user-provided HS code hints treated as
                unverified retrieval terms.

        Returns:
            Ordered list of unique keywords for hybrid retrieval.
        """

        keywords = list(facts.keywords)

        attributes = facts.main_attributes

        keywords.extend(
            value
            for value in [
                facts.product_category,
                attributes.product_type,
                attributes.material,
            ]
            if value
        )

        if attributes.is_part:
            keywords.append("part")

        if user_hs_codes:
            keywords.extend(user_hs_codes)

        return list(
            dict.fromkeys(
                keyword.strip() for keyword in keywords if keyword and keyword.strip()
            )
        )
