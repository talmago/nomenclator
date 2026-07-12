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
    - Use chapter notes and heading descriptions as primary evidence.
    - Apply HS legal classification principles.
    - Prefer specific classifications over general headings.
    - Compare competing classifications when ambiguity exists.
    - Explain why the selected candidates are preferred.

    Do not:
    - Return chapter numbers as final classifications.
    - Return 4-digit headings as final classifications.
    - Classify based only on keyword similarity.
    - Assume the highest-ranked retrieved chapter is correct.
    - Ignore exclusions or legal notes.

    Output:
    - Return ranked 6-digit HS subheading candidates.
    - Include clear legal reasoning for each candidate.
    """

    product_facts: ProductFactsModel = dspy.InputField(
        desc="Structured product facts extracted from the product description."
    )

    chapter_context: list[dict] = dspy.InputField(
        desc=(
            "Relevant HS chapter context including chapter metadata, notes, "
            "and heading hierarchy."
        )
    )

    classification: HSClassificationOutputModel = dspy.OutputField(
        desc="Ranked 6-digit HS subheading candidates with legal reasoning."
    )
