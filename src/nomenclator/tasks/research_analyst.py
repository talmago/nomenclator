import dspy

from nomenclator.models.navigation import (
    HSResearchContext,
    HSResearchSelectionOutputModel,
)
from nomenclator.models.product_facts import ProductFactsModel


class ResearchAnalystSignature(dspy.Signature):
    """Select the most plausible HS chapters for detailed classification.

    You are an HS Research Analyst specializing in HS 2022 nomenclature.

    Analyze the structured product facts together with the retrieved HS context
    and select the chapter pathways that the Classification Analyst should
    examine next.

    The retrieved context is produced by a high-recall hybrid search. It may
    contain irrelevant chapters and may not contain the ideal chapter.
    Retrieval rank and keyword similarity are weak signals only.

    Base your analysis primarily on:

    - the product's essential nature;
    - function, material, composition, and other product facts;
    - chapter and section scope;
    - legally relevant section notes, definitions, and exclusions;
    - whether the chapter describes the product itself rather than merely its
      application, host equipment, or end use.

    Eliminate clearly incompatible chapters and rank the remaining pathways
    from most to least plausible.

    Select the best available pathways from the provided context even when none
    appears definitive. Return no candidates only when every retrieved chapter
    is clearly incompatible with the product.

    Do not:

    - assign an HS code below chapter level;
    - make the final classification decision;
    - apply the General Rules for Interpretation;
    - invent chapter references;
    - return chapters that are only weakly related by terminology or end use.

    Output requirements:

    - Return only chapter references present in research_context.
    - Return at most ``max_candidates`` results.
    - Rank results from most to least plausible.
    - Return only chapter references, without titles, scores, or reasoning.
    """

    product_facts: ProductFactsModel = dspy.InputField(
        desc="Structured product facts extracted by the Product Analyst."
    )

    research_context: HSResearchContext = dspy.InputField(
        desc=(
            "Structured HS research context containing candidate chapters "
            "grouped by section, including relevant section notes."
        )
    )

    max_candidates: int = dspy.InputField(
        desc=(
            "Maximum number of chapter pathways to return. "
            "Return fewer when fewer are sufficiently supported."
        )
    )

    navigation: HSResearchSelectionOutputModel = dspy.OutputField(
        desc=("Ranked chapter references selected from research_context.")
    )


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
            max_candidates: Maximum number of chapter references to return.
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
    ) -> HSResearchSelectionOutputModel:
        """Rank relevant HS chapter pathways.

        Args:
            product_facts: Structured product facts.
            research_context: Retrieved HS chapter and section context.

        Returns:
            Minimal ranked chapter selections.
        """

        result = self.analyze(
            product_facts=product_facts,
            research_context=research_context,
            max_candidates=self._max_candidates,
        )

        return result.navigation
