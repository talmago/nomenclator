import dspy

from nomenclator.models.classification import HSClassificationOutputModel
from nomenclator.models.product_facts import ProductFactsModel


class ClassificationAnalystSignature(dspy.Signature):
    """Classify products according to the Harmonized System nomenclature.

    You are a Customs Specialist with expertise in HS classification.

    Determine the most appropriate HS 6-digit subheading using the product
    facts and relevant HS chapter information.

    The provided chapters were selected by a previous research stage. They are
    possible classification pathways, not confirmed classifications.

    You must:
    - Analyze the product facts together with the provided HS chapters.
    - Identify the correct heading and 6-digit subheading.
    - Use chapter notes and heading descriptions as primary evidence for
      scope and coverage.
    - Apply the provided General Rules for the Interpretation of the
      Harmonized System (GIR) when resolving conflicts, incomplete or
      unfinished goods, mixtures, composite goods, and packing.
    - Prefer specific classifications over general headings.
    - Compare competing classifications when ambiguity exists.
    - Explain why the selected candidates are preferred, citing relevant
      GIR rules where they affect the decision.

    Do not:
    - Return chapter numbers as final classifications.
    - Return 4-digit headings as final classifications.
    - Classify based only on keyword similarity.
    - Assume the highest-ranked retrieved chapter is correct.
    - Ignore exclusions, legal notes, or applicable GIR rules.

    Scoring:
    - Assign each candidate a confidence score between 0.0 and 1.0, where 1.0
      means highest confidence in the classification.
    - Scores must never fall outside the 0.0-1.0 range.

    Output:
    - Return up to max_candidates ranked 6-digit HS subheading candidates.
    - When competing classification pathways are plausible, include them as
      additional ranked candidates (even at lower confidence) so the strongest
      alternatives remain visible. Return fewer than max_candidates only when a
      single classification is unambiguously correct.
    - Include clear legal reasoning for each candidate.
    """

    product_facts: ProductFactsModel = dspy.InputField(
        desc="Structured product facts extracted from the product description."
    )

    general_rules: list[dict] = dspy.InputField(
        desc=(
            "General Rules for the Interpretation of the Harmonized System "
            "(GIR). Each entry has rule id and full rule text."
        )
    )

    chapter_context: list[dict] = dspy.InputField(
        desc=(
            "Relevant HS chapter context including chapter metadata, notes, "
            "and heading hierarchy."
        )
    )

    max_candidates: int = dspy.InputField(
        desc=(
            "Maximum number of ranked HS subheading candidates to return. "
            "Include competing alternatives up to this limit; return fewer "
            "only when a single classification is unambiguously correct."
        )
    )

    classification: HSClassificationOutputModel = dspy.OutputField(
        desc=(
            "Ranked 6-digit HS subheading candidates with legal reasoning and "
            "confidence scores between 0.0 and 1.0. Candidates must not exceed "
            "max_candidates."
        )
    )


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
        general_rules: list[dict],
        max_candidates: int,
    ) -> HSClassificationOutputModel:
        """Classify a product.

        Args:
            product_facts: Structured product facts extracted from the product
                description.
            chapter_context: Relevant HS chapter context including notes and
                heading hierarchy.
            general_rules: Compact GIR entries (rule id + text) used for legal
                interpretation.
            max_candidates: Maximum number of ranked HS subheading candidates to
                return.

        Returns:
            Ranked 6-digit HS code candidates.
        """

        result = self.classify(
            product_facts=product_facts,
            chapter_context=chapter_context,
            general_rules=general_rules,
            max_candidates=max_candidates,
        )

        return result.classification
