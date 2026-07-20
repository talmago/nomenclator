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
| HS Navigation Document Builder |
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
+-------------------------+
| Research Context Builder|
+-------------------------+
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
HSResearchSelectionOutputModel
        |
        v
+--------------------------------+
| Classification Context Builder |
+--------------------------------+
        |
        v
 HSClassificationContext
        |
        v
+-------------------------+
| Classification Analyst  |
+-------------------------+
        |
        v
HSClassificationSelectionOutputModel
        |
        v
+------------------------------+
| Classification Result Builder|
+------------------------------+
        |
        v
HSClassificationOutputModel
```

The Product Analyst extracts structured product facts from the raw product
description.

The HS Navigation Document Builder converts the parsed HS nomenclature tree
into searchable documents. The Hybrid Retriever performs deterministic
semantic and lexical retrieval to identify potentially relevant HS chapters.

The Research Context Builder groups retrieved chapters by HS section and
enriches them with the corresponding section notes, providing legal context
for pathway selection without loading full chapter documents.

The Research Analyst evaluates the retrieved context and selects the most
plausible HS chapters for detailed analysis.

The Classification Context Builder loads only the shortlisted chapters,
retrieves the most relevant heading hierarchy for each chapter, combines it
with the General Rules for the Interpretation of the Harmonized System (GIR),
and produces a compact classification context.

The Classification Analyst performs detailed legal reasoning over this context
and returns ranked HS code selections together with confidence scores and
supporting reasoning.

Finally, the Classification Result Builder enriches the selected HS codes with
their canonical descriptions and source chapters from the classification
context to produce the final structured output.

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
    HSClassificationChapterContext,
    HSClassificationContext,
    HSClassificationHeadingContext,
    HSClassificationOutputModel,
    HSClassificationSelectionOutputModel,
    HSClassificationSubheadingContext,
    HSCodeCandidateModel,
)
from nomenclator.models.navigation import (
    HSResearchChapterContext,
    HSResearchContext,
    HSResearchSectionContext,
    HSResearchSelectionOutputModel,
)
from nomenclator.models.product_facts import ProductFactsModel
from nomenclator.nomenclature.chapter import HSChapter
from nomenclator.nomenclature.client import NomenclatureClient
from nomenclator.nomenclature.parser import _chapter_ref_from_number
from nomenclator.nomenclature.tree import HSDocumentRef, HSHeading
from nomenclator.nomenclature.urls import chapter_url_from_ref
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
        retrieved: Raw chapter retrieval results from the Nomenclature
            Retriever (semantic + BM25 hybrid search).
        navigation: Ranked chapter candidates produced by the Research
            Analyst.
        classification: Final ranked HS code candidates produced by the
            Classification Analyst.
    """

    facts: ProductFactsModel
    retrieved: list[SearchResult[HSDocumentRef]]
    navigation: HSResearchSelectionOutputModel
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
        Navigation Retriever
                |
                v
        Candidate HS chapters
                |
                v
        Research Context Builder
                |
                v
        Research Analyst
                |
                v
        Selected chapter references
                |
                v
        Classification Context Builder
                |
                v
        Classification Analyst
                |
                v
        Ranked HS code candidates

    The pipeline combines deterministic retrieval with LLM reasoning.

    The navigation retriever performs high-recall semantic and lexical search
    over the HS nomenclature to identify potentially relevant chapters. The
    Research Analyst then eliminates irrelevant pathways and selects the most
    plausible chapters for further analysis.

    The Classification Context Builder retrieves the most relevant headings from
    the selected chapters, always preserving chapter notes while keeping the
    prompt within a configurable context budget.

    Finally, the Classification Analyst identifies the most plausible HS codes
    from the provided context. The application then resolves the selected codes
    into the canonical classification output.
    """

    def __init__(
        self,
        *,
        client: NomenclatureClient | None = None,
        embedding_model: str = "minishlab/potion-base-8M",
        max_retrieved_chapters: int = 5,
        max_research_chapters: int = 3,
        max_classification_chunks: int = 10,
    ) -> None:
        """Initialize the classification agent.

        Args:
            client:
                HS nomenclature client.

            model_name:
                Embedding model used by the retrievers.

            max_retrieved_chapters:
                Maximum number of chapters retrieved and passed to the Research
                Analyst.

            max_research_chapters:
                Maximum number of chapter pathways shortlisted by the Research
                Analyst for classification analysis.

            max_classification_chunks:
                Global maximum number of heading-level chunks included in the
                Classification Analyst context across all shortlisted chapters.
        """
        if max_retrieved_chapters <= 0:
            raise ValueError("max_retrieved_chapters must be greater than zero")

        if max_research_chapters <= 0:
            raise ValueError("max_candidates must be greater than zero")

        if max_classification_chunks <= 0:
            raise ValueError("max_chunks must be greater than zero")

        self._client = client or NomenclatureClient()
        self._embedding_model = embedding_model
        self._max_retrieved_chapters = max_retrieved_chapters
        self._max_research_chapters = max_research_chapters
        self._max_classification_chunks = max_classification_chunks

        try:
            self._tree = self._client.get_tree()
        except Exception as exc:
            raise HSInitializationError(
                "Failed to initialize HS classification resources"
            ) from exc

        self._product_analyst = ProductAnalyst()
        self._research_analyst = ResearchAnalyst(max_candidates=max_research_chapters)
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
            from each upstream stage (facts, retrieved chapters and navigation).

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

        retrieved = self._retrieve_chapters(facts)

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

        if not navigation.candidates:
            return HSClassificationResult(
                facts=facts,
                retrieved=retrieved,
                navigation=navigation,
                classification=HSClassificationOutputModel(candidates=[]),
            )

        # --------------------------------------------------
        # Step 4: Classification Analyst
        # --------------------------------------------------

        try:
            classification_context = self._build_classification_context(
                navigation,
                facts=facts,
            )
        except Exception as exc:
            raise HSClassificationPipelineError(
                "Failed to load HS classification context"
            ) from exc

        try:
            selection = self._classification_analyst(
                facts,
                classification_context,
            )

            classification = self._build_classification_result(
                selection, classification_context
            )
        except Exception as exc:
            raise HSClassificationAnalysisError(
                "Failed to produce HS classification candidates"
            ) from exc

        return HSClassificationResult(
            facts=facts,
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
        navigation: HSResearchSelectionOutputModel,
        *,
        facts: ProductFactsModel,
    ) -> HSClassificationContext:
        """Build the Classification Analyst context.

        Resolves the chapter references selected by the Research Analyst into
        canonical HS chapter documents, retrieves the most relevant heading
        hierarchy for each shortlisted chapter, and combines the resulting chapter
        context with the General Rules for the Interpretation of the Harmonized
        System (GIR).

        The retrieved heading hierarchy is globally ranked across all shortlisted
        chapters while guaranteeing that each shortlisted chapter contributes at
        least one heading, when available. Chapter notes are always included because
        they provide legally binding context for classification.

        Args:
            navigation: Ranked chapter references selected by the Research Analyst.
            facts: Structured product facts used as the heading retrieval query.

        Returns:
            Complete legal and structural context for the Classification Analyst.
        """

        chapters = [
            self._client.get_chapter(candidate.chapter_ref)
            for candidate in navigation.candidates
        ]

        headings_by_chapter = self._retrieve_headings(chapters, facts)

        return HSClassificationContext(
            general_rules=[
                rule.to_dict() for rule in self._client.get_general_rules().rules
            ],
            chapters=[
                HSClassificationChapterContext(
                    chapter_number=chapter.chapter_number,
                    title=chapter.title,
                    notes=[note.to_dict() for note in chapter.notes],
                    headings=[
                        HSClassificationHeadingContext(
                            code=heading.code,
                            description=heading.description,
                            subheadings=[
                                HSClassificationSubheadingContext(
                                    code=subheading.code,
                                    description=subheading.description,
                                )
                                for subheading in heading.subheadings
                            ],
                        )
                        for heading in headings_by_chapter.get(
                            chapter.chapter_number,
                            [],
                        )
                    ],
                )
                for chapter in chapters
            ],
        )

    def _build_classification_result(
        self,
        selection: HSClassificationSelectionOutputModel,
        context: HSClassificationContext,
    ) -> HSClassificationOutputModel:
        """Build the final HS classification result.

        Resolves the minimal classification candidates produced by the
        Classification Analyst into the canonical output model by enriching each
        selected HS code with its description, source chapter, and chapter URL.

        Args:
            selection: Ranked HS code candidates returned by the Classification
                Analyst.
            context: HS classification context used during classification.

        Returns:
            Hydrated HS classification result.
        """

        subheadings = {
            subheading.code: (subheading, heading, chapter)
            for chapter in context.chapters
            for heading in chapter.headings
            for subheading in heading.subheadings
        }

        candidates: list[HSCodeCandidateModel] = []

        for candidate in selection.candidates:
            subheading, heading, chapter = subheadings[candidate.code]

            heading_description = heading.description.rstrip(" .:")
            subheading_description = subheading.description.strip()

            ref = _chapter_ref_from_number(chapter.chapter_number)
            source_url = chapter_url_from_ref(ref)

            candidates.append(
                HSCodeCandidateModel(
                    code=subheading.code,
                    description=(f"{heading_description} — {subheading_description}"),
                    score=candidate.confidence,
                    reasoning=candidate.reasoning,
                    source_chapter=ref,
                    source_url=source_url,
                )
            )

        return HSClassificationOutputModel(
            candidates=candidates,
        )

    def _retrieve_chapters(
        self,
        product_facts: ProductFactsModel,
    ) -> list[SearchResult[HSDocumentRef]]:
        """Retrieve candidate HS chapters for a product.

        Builds the HS navigation retriever inline over the parsed nomenclature
        tree and searches it using the retrieval query derived from the extracted
        product facts. The retriever is constructed on demand rather than held as
        agent state, since it is only used at this stage of the pipeline.

        Args:
            product_facts: Structured product facts extracted by the Product
                Analyst.

        Returns:
            Ranked retrieval results whose payloads are HS chapter references.
        """

        documents = [
            RetrievalDocument(
                id=chapter.ref,
                content=(
                    f"{section.label}\n"
                    f"{section.title}\n\n"
                    f"Chapter {chapter.ref}\n"
                    f"{chapter.title}"
                ),
                payload=chapter,
            )
            for section in self._tree.sections
            for chapter in section.chapters
            if chapter.ref
        ]

        retriever = Retriever(
            documents,
            model_name=self._embedding_model,
        )

        return retriever.search(
            product_facts.retrieval_query(),
            limit=self._max_retrieved_chapters,
        )

    def _retrieve_headings(
        self,
        chapters: list[HSChapter],
        product_facts: ProductFactsModel,
    ) -> dict[int, list[HSHeading]]:
        """Retrieve the most relevant headings across all shortlisted chapters.

        A single hybrid retriever is built over the heading chunks of every
        shortlisted chapter and searched once using the retrieval query derived
        from the product facts. The retriever returns a global ranking over all
        chunks. Selection then guarantees one chunk per chapter that has headings
        (the floor), and fills the remaining budget with the next most relevant
        chunks overall.

        Args:
            chapters: Parsed shortlisted chapters to search within.
            product_facts: Structured product facts extracted by the Product
                Analyst.

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
            model_name=self._embedding_model,
        )

        results = retriever.search(
            product_facts.retrieval_query(),
            limit=len(all_chunks),
        )

        return self._select_headings(
            results,
            chunk_to_chapter=chunk_to_chapter,
            chapters_with_chunks=chapters_with_chunks,
            budget=self._max_classification_chunks,
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
