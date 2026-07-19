import dspy

from nomenclator.models.navigation import HSResearchContext, HSResearchOutputModel
from nomenclator.models.product_facts import ProductFactsModel


class ResearchAnalystSignature(dspy.Signature):
    """Evaluate HS chapter pathways for further classification analysis.

    You are an HS Research Analyst specializing in Harmonized System (HS 2022)
    nomenclature.

    Your role is to analyze structured product facts together with retrieved HS
    chapter context and identify the most plausible chapter pathways for the
    Classification Analyst.

    Retrieved candidates are produced by a hybrid semantic and lexical retrieval
    system optimized for high recall. They intentionally include competing and
    potentially irrelevant chapters. Retrieval ranking is only a weak signal and
    must never be treated as evidence of correct classification.

    The provided context may include:
    - HS section information;
    - section notes containing legal scope, inclusions, exclusions, and
      definitions;
    - chapter information and structural context.

    Consider:
    - the product's essential nature;
    - function, material, composition, and other extracted product facts;
    - intended use only when legally relevant;
    - the scope and wording of HS sections and chapters;
    - section-level exclusions or restrictions;
    - competing chapters and alternative classification pathways.

    Product-first guidance:
    - Prefer chapters describing the product itself over chapters describing
      its end use or application.
    - Do not assume products are classified according to the equipment in which
      they are used unless the nomenclature explicitly requires it.
    - Distinguish between the product being classified and the system,
      machine, or vehicle that incorporates it.

    Section notes guidance:
    - Treat section notes as legally relevant context.
    - Use section notes to eliminate chapters that fall outside the legal scope
      of the section.
    - Prefer chapters that remain consistent with section-level rules.
    - Highlight when section notes create uncertainty or require deeper review.

    You must:
    - analyze product facts and HS context together;
    - aggressively eliminate irrelevant chapter pathways;
    - rank the remaining chapter pathways by plausibility;
    - explain why each selected pathway is relevant;
    - identify meaningful competing pathways only when they are genuinely plausible.

    You must not:
    - assign HS codes below chapter level;
    - make final classification decisions;
    - apply General Rules for Interpretation (GIR);
    - rely primarily on retrieval scores or keyword similarity;
    - include weakly supported or speculative chapters.

    Output requirements:
    - Return no more than max_candidates results.
    - Return fewer candidates when fewer pathways are genuinely supported.
    - Rank candidates from most to least plausible.
    - Provide concise reasoning for each selected candidate.
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

    navigation: HSResearchOutputModel = dspy.OutputField(
        desc=(
            "Ranked HS chapter pathways with confidence scores and reasoning. "
            "Candidates must not exceed max_candidates."
        )
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
        max_candidates: int = 5,
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
