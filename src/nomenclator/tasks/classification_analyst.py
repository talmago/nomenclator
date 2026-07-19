import dspy

from nomenclator.models.classification import HSClassificationOutputModel
from nomenclator.models.product_facts import ProductFactsModel


class ClassificationAnalystSignature(dspy.Signature):
    """Classify products according to the Harmonized System nomenclature.

    You are a Customs Specialist with expertise in HS classification.

    Determine the most appropriate HS 6-digit subheading using the product
    facts and relevant HS chapter information.

    The provided chapters and headings were selected by previous retrieval and
    research stages. They represent possible classification pathways, not
    confirmed classifications.

    You must:
    - Analyze the product facts together with the provided HS context.
    - Identify the correct heading and 6-digit subheading.
    - Use chapter notes and heading descriptions as primary evidence for scope
      and coverage.
    - Apply the provided General Rules for the Interpretation of the
      Harmonized System (GIR) when resolving conflicts, incomplete or
      unfinished goods, mixtures, composite goods, and packing.
    - Prefer specific classifications over general headings.
    - Compare genuinely plausible competing classifications.
    - Explain why each candidate is supported and why the preferred candidate
      ranks above the alternatives.
    - Cite relevant GIR rules when they affect the decision.

    You must not:
    - Return chapter numbers or 4-digit headings as final classifications.
    - Classify based only on keyword similarity.
    - Assume the highest-ranked retrieved chapter is correct.
    - Ignore exclusions, legal notes, or applicable GIR rules.
    - Include weakly supported or speculative alternatives.

    Scoring:
    - Assign each candidate a confidence score between 0.0 and 1.0.
    - Scores must reflect relative confidence and must not exceed that range.

    Output:
    - Return ranked 6-digit HS subheading candidates.
    - Return the strongest supported classification first.
    - Include additional candidates only when a meaningful competing
      classification remains plausible.
    - Return a single candidate when the classification is unambiguous.
    - Include concise legal reasoning for each candidate.
    """

    product_facts: ProductFactsModel = dspy.InputField(
        desc="Structured product facts extracted from the product description."
    )

    general_rules: list[dict] = dspy.InputField(
        desc=(
            "General Rules for the Interpretation of the Harmonized System "
            "(GIR). Each entry contains the rule identifier and full text."
        )
    )

    chapter_context: list[dict] = dspy.InputField(
        desc=(
            "Relevant HS chapter context containing chapter metadata, legal "
            "notes, and the retrieved heading hierarchy."
        )
    )

    classification: HSClassificationOutputModel = dspy.OutputField(
        desc=(
            "Ranked 6-digit HS subheading candidates with concise legal "
            "reasoning and confidence scores between 0.0 and 1.0."
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
    ) -> HSClassificationOutputModel:
        """Classify a product.

        Args:
            product_facts: Structured product facts extracted from the product
                description.
            chapter_context: Relevant HS chapter context including notes and
                heading hierarchy.
            general_rules: Compact GIR entries (rule id + text) used for legal
                interpretation.

        Returns:
            Ranked 6-digit HS code candidates.
        """

        result = self.classify(
            product_facts=product_facts,
            chapter_context=chapter_context,
            general_rules=general_rules,
        )

        return result.classification
