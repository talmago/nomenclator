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

The Classification Analyst loads only the shortlisted chapter documents and
performs detailed legal classification to produce ranked HS code candidates.

The retrieval layer is deterministic and independent from LLM reasoning,
allowing retrieval quality and classification reasoning to be evaluated
separately.
"""

from __future__ import annotations

import dspy

from nomenclator.exceptions import (
    HSClassificationPipelineError,
    HSInitializationError,
    HSNoCandidatesFoundError,
)
from nomenclator.models.classification import HSClassificationOutputModel
from nomenclator.models.navigation import (
    HSResearchChapterContext,
    HSResearchContext,
    HSResearchOutputModel,
    HSResearchSectionContext,
)
from nomenclator.models.product_facts import ProductFactsModel
from nomenclator.models.search import RetrievalDocument, SearchResult
from nomenclator.models.tree import HSDocumentRef
from nomenclator.nomenclature.client import NomenclatureClient
from nomenclator.retrieval import Retriever
from nomenclator.signatures.classification_analyst import ClassificationAnalystSignature
from nomenclator.signatures.product_analyst import ProductAnalystSignature
from nomenclator.signatures.research_analyst import ResearchAnalystSignature


class ProductAnalyst(dspy.Module):
    """Extract structured product facts for HS classification.

    The Product Analyst transforms a raw product description into structured
    information used by downstream retrieval and classification stages.

    It does not perform HS classification.
    """

    def __init__(self) -> None:
        """Initialize the Product Analyst."""

        super().__init__()

        self.extract = dspy.Predict(
            ProductAnalystSignature,
        )

    def forward(
        self,
        description: str,
        hints: list[str] | None = None,
    ) -> ProductFactsModel:
        """Extract product facts.

        Args:
            description: Raw product description.
            hints: Optional user-provided HS code hints.

        Returns:
            Structured product facts.
        """

        result = self.extract(
            description=description,
            hints=hints or [],
        )

        return result.product_facts


class ResearchAnalyst(dspy.Module):
    """Analyze retrieved HS chapter candidates.

    The Research Analyst evaluates retrieval results and produces a ranked
    shortlist of relevant HS chapter candidates for downstream classification.

    The module does not assign HS codes. It only identifies plausible chapter
    pathways based on product facts and retrieved nomenclature context.
    """

    def __init__(
        self,
        *,
        max_candidates: int = 3,
    ) -> None:
        """Initialize the Research Analyst.

        Args:
            max_candidates: Maximum number of relevant HS chapter candidates
                to return.
        """

        super().__init__()

        self._max_candidates = max_candidates

        self.analyze = dspy.Predict(
            ResearchAnalystSignature,
        )

    def forward(
        self,
        product_facts: ProductFactsModel,
        research_context: HSResearchContext,
    ) -> HSResearchOutputModel:
        """Rank relevant HS chapter pathways.

        Args:
            product_facts: Structured product facts extracted from the product
                description.
            research_context: HS legal context containing retrieved candidate
                chapters grouped by section, including section notes.

        Returns:
            Ranked HS chapter pathways for downstream classification.
        """

        result = self.analyze(
            product_facts=product_facts,
            research_context=research_context,
            max_candidates=self._max_candidates,
        )

        return result.navigation


class ClassificationAnalyst(dspy.Module):
    """Produce HS classification candidates using legal reasoning."""

    def __init__(self) -> None:
        """Initialize the Classification Analyst."""

        super().__init__()

        self.classify = dspy.ChainOfThought(
            ClassificationAnalystSignature,
        )

    def forward(
        self,
        product_facts: ProductFactsModel,
        chapter_context: list[dict],
    ) -> HSClassificationOutputModel:
        """Classify a product.

        Args:
            product_facts: Structured product facts extracted from the product
                description.
            chapter_context: Relevant HS chapter context including notes and
                heading hierarchy.

        Returns:
            Ranked 6-digit HS code candidates.
        """

        result = self.classify(
            product_facts=product_facts,
            chapter_context=chapter_context,
        )

        return result.classification


class HSProductAnalysisError(HSClassificationPipelineError):
    """Raised when product analysis fails."""


class HSResearchAnalysisError(HSClassificationPipelineError):
    """Raised when HS research analysis fails."""


class HSClassificationAnalysisError(HSClassificationPipelineError):
    """Raised when final classification analysis fails."""


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
        Classification Analyst
                |
                v
        HS code candidates

    The navigation retrieval layer is deterministic and uses semantic and lexical
    search over HS nomenclature documents. It is independent from LLM reasoning.

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
    ) -> None:
        """Initialize the classification agent.

        Args:
            client: HS nomenclature client.
            model_name: Embedding model used by the retriever.
            retrieval_limit: Maximum number of HS chapters retrieved by the hybrid
                retriever and passed to the research analyst.
            max_candidates: Maximum number of relevant HS chapter candidates
                returned by the research analyst for downstream classification.
        """
        if retrieval_limit <= 0:
            raise ValueError("retrieval_limit must be greater than zero")

        if max_candidates <= 0:
            raise ValueError("max_candidates must be greater than zero")

        self._client = client or NomenclatureClient()
        self._retrieval_limit = retrieval_limit

        try:
            self._tree = self._client.get_tree()
            self._retriever = self._build_hs_navigation_retriever(model_name)
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
    ) -> HSClassificationOutputModel:
        """Classify a product description.

        Args:
            description: Raw product description.
            user_hs_codes: Optional user-provided HS hints.

        Returns:
            Final HS classification candidates.
        """

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

        retrieved = self._retriever.search(
            facts.normalized_description,
            keywords=keywords,
            limit=self._retrieval_limit,
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
            chapter_context = [
                self._client.get_chapter(
                    candidate.chapter_ref
                ).to_classification_context()
                for candidate in navigation.candidates
            ]
        except Exception as exc:
            raise HSClassificationPipelineError(
                "Failed to load HS chapter context"
            ) from exc

        try:
            classification = self._classification_analyst(
                facts,
                chapter_context,
            )
        except Exception as exc:
            raise HSClassificationAnalysisError(
                "Failed to produce HS classification candidates"
            ) from exc

        return classification

    def _build_hs_navigation_retriever(
        self, model_name: str
    ) -> Retriever[HSDocumentRef]:
        """Build the retriever used for HS chapter navigation.

        The parsed HS nomenclature tree is converted into retrieval documents and
        indexed by a hybrid retriever that combines semantic and lexical search.
        The resulting retriever is used to identify candidate chapters for
        downstream classification.

        Args:
            model_name: Optional embedding model identifier.

        Returns:
            A configured hybrid retriever over HS chapter documents.
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

        return Retriever(
            documents,
            model_name=model_name,
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

    def _retrieval_keywords(
        self,
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
