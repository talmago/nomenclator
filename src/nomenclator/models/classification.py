from pydantic import BaseModel


class HSCodeCandidateModel(BaseModel):
    """Candidate HS code produced by the classifier."""

    code: str
    description: str
    score: float
    reasoning: list[str]
    source_chapter: str


class HSClassificationOutputModel(BaseModel):
    """Structured output of the HS classifier."""

    candidates: list[HSCodeCandidateModel]
