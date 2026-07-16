from dataclasses import dataclass

from nomenclator.models.classification import (
    HSClassificationOutputModel,
    HSCodeCandidateModel,
)
from nomenclator.models.navigation import HSResearchOutputModel
from nomenclator.models.product_facts import ProductFactsModel
from nomenclator.models.search import SearchResult
from nomenclator.models.tree import HSDocumentRef


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
