import dspy

from nomenclator.models.classification import (
    HSClassificationContext,
    HSClassificationSelectionOutputModel,
)
from nomenclator.models.product_facts import ProductFactsModel


class ClassificationAnalystSignature(dspy.Signature):
    """Classify products according to the Harmonized System (HS 2022).

    You are a Customs Specialist with expertise in HS classification.

    Determine the most appropriate 6-digit HS subheading using the structured
    product facts and the provided HS classification context.

    The Research Analyst has already narrowed the search space. The provided
    context contains the best available classification pathways and is expected
    to include at least one applicable HS subheading.

    The context contains:

    - the General Rules for the Interpretation of the Harmonized System (GIR);
    - shortlisted chapters;
    - chapter notes;
    - relevant headings and 6-digit subheadings.

    Your task is to evaluate the HS subheadings in the provided context and rank
    them according to how well they classify the product.

    You must:

    - Select a primary 6-digit HS code from the provided context.
    - Base the classification on the product facts and the legal scope of the
      provided headings, subheadings, and notes.
    - Consider the product's nature, function, material, composition, and
      essential character where relevant.
    - Apply the GIR when they affect the classification.
    - Prefer the most specific applicable subheading.
    - Compare competing pathways when more than one classification is
      reasonably supported.
    - Explain why the primary candidate is preferred over any alternatives.

    Every returned code must exactly match a 6-digit HS subheading present in
    the provided classification context.

    Output:

    - Return at least one ranked candidate.
    - Return the strongest supported candidate first.
    - Include additional candidates only when another classification is
      reasonably supported by the product facts and HS context.
    - Return a single candidate when one classification is clearly preferred.
    - Assign each candidate a confidence score between 0.0 and 1.0.
    - Include concise classification reasoning for each candidate.
    """

    product_facts: ProductFactsModel = dspy.InputField(
        desc="Structured product facts extracted from the product description."
    )

    context: HSClassificationContext = dspy.InputField(
        desc=(
            "HS classification context containing the General Rules for the "
            "Interpretation of the Harmonized System (GIR), shortlisted "
            "chapters, legal notes, headings, and candidate subheadings."
        )
    )

    classification: HSClassificationSelectionOutputModel = dspy.OutputField(
        desc=(
            "Ranked 6-digit HS subheading references with confidence scores "
            "and concise legal reasoning."
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
        context: HSClassificationContext,
    ) -> HSClassificationSelectionOutputModel:
        """Classify a product.

        Args:
            product_facts: Structured product facts extracted from the product
                description.
            context: HS classification context containing the General Rules for
                the Interpretation of the Harmonized System (GIR), shortlisted
                chapters, legal notes, headings, and candidate subheadings.

        Returns:
            Ranked 6-digit HS subheading candidates.
        """

        result = self.classify(
            product_facts=product_facts,
            context=context,
        )

        return result.classification
