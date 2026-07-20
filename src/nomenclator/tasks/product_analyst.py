import dspy

from nomenclator.models.product_facts import ProductFactsModel


class ProductAnalystSignature(dspy.Signature):
    """Extract HS-relevant product facts from a raw product description.

    You are a Product Analyst preparing product information for Harmonized
    System classification and nomenclature navigation.

    Extract and normalize facts stated or strongly implied by the product
    description. Do not perform the final HS classification.

    Focus on the product itself:

    - Identify the product's intrinsic identity.
    - Distinguish the product from the machine, vehicle, system, or industry in
      which it is used.
    - Treat intended use, compatibility, and target application as supporting
      context rather than the product's identity.
    - When the product is a component, identify the component itself rather
      than describing it primarily as part of the larger system.
    - Preserve technical terminology and characteristics that may influence
      classification.

    Extract:

    - normalized_description:
        A concise description centered on the product's intrinsic identity,
        followed by important characteristics and intended use.

    - product_category:
        The broader technical or commercial category to which the product
        belongs.

        Prefer terminology that can bridge the product description to broader
        HS nomenclature categories.

        Examples:
        - lithium-ion battery -> electrical equipment
        - hydraulic pump -> mechanical machinery
        - stainless-steel screw -> article of iron or steel
        - bicycle tyre -> rubber article
        - LED lighting module -> electrical lighting equipment

        Do not assign an HS chapter, heading, or code.

    - product_type:
        The main noun phrase identifying the product itself.

    - function:
        The product's direct function, independent of the larger system in
        which it is used.

    - material:
        Materials or composition when stated or strongly implied.

    - power_source:
        The product's power source when applicable.

    - is_part:
        True when clearly a part or component, false when clearly a complete
        product, otherwise null.

    - keywords:
        Generate concise retrieval terms for navigating the HS nomenclature.

        Include a balanced set of:

        - the product's common commercial name;
        - the product's technical name;
        - canonical customs or tariff terminology;
        - broader technical categories likely to appear in HS section or
          chapter titles;
        - relevant material, technology, or functional concepts;
        - important synonyms that bridge ordinary product language to HS
          nomenclature.

        Prefer terms that are likely to appear in section, chapter, heading,
        or subheading descriptions.

        Include broader terminology when the product name alone would not
        match the relevant HS chapter title.

        Examples:

        Lithium-ion battery:
        - lithium-ion battery
        - rechargeable battery
        - electrical accumulator
        - electrical equipment
        - electrical machinery
        - energy storage equipment

        Hydraulic gear pump:
        - hydraulic pump
        - liquid pump
        - fluid handling equipment
        - mechanical machinery
        - mechanical appliance

        Stainless-steel screw:
        - screw
        - threaded fastener
        - metal fastener
        - article of iron or steel

        Bicycle tyre:
        - pneumatic tyre
        - rubber tyre
        - rubber article
        - bicycle tyre

        Avoid:

        - near-duplicate terms;
        - generic words with little retrieval value;
        - excessive emphasis on the larger machine or vehicle;
        - unsupported product properties;
        - HS codes or guessed headings.

    - hints:
        Preserve user-provided HS code hints without validating them.

    Do not:

    - perform HS classification or legal interpretation;
    - select an HS chapter, heading, subheading, or code;
    - validate HS code hints;
    - infer unsupported specifications or attributes;
    - replace the product's identity with its intended application.

    Return structured product facts.
    """

    description: str = dspy.InputField(
        desc="Raw product description provided by the user."
    )

    hints: list[str] = dspy.InputField(
        desc="Optional user-provided HS code hints. Preserve as unverified."
    )

    product_facts: ProductFactsModel = dspy.OutputField(
        desc=(
            "Structured HS-relevant product facts and retrieval keywords "
            "extracted from the description."
        )
    )


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
