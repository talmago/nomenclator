from pydantic import BaseModel


class HSClassificationSubheadingContext(BaseModel):
    """HS subheading available for classification."""

    code: str
    description: str


class HSClassificationHeadingContext(BaseModel):
    """HS heading available for classification."""

    code: str
    description: str
    subheadings: list[HSClassificationSubheadingContext]


class HSClassificationChapterContext(BaseModel):
    """Classification context for a shortlisted chapter."""

    chapter_number: int
    title: str
    notes: list[dict]
    headings: list[HSClassificationHeadingContext]


class HSClassificationContext(BaseModel):
    """HS context provided to the Classification Analyst."""

    general_rules: list[dict]
    chapters: list[HSClassificationChapterContext]


class HSClassificationSelectionModel(BaseModel):
    """Minimal HS classification candidate."""

    code: str
    confidence: float
    reasoning: list[str]


class HSClassificationSelectionOutputModel(BaseModel):
    """Classification Analyst output."""

    candidates: list[HSClassificationSelectionModel]


class HSCodeCandidateModel(BaseModel):
    code: str
    description: str
    score: float
    reasoning: list[str]
    source_chapter: str
    source_url: str | None = None


class HSClassificationOutputModel(BaseModel):
    """Structured output of the HS classifier."""

    candidates: list[HSCodeCandidateModel]
