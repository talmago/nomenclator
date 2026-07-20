from pydantic import BaseModel


class HSResearchChapterContext(BaseModel):
    """Chapter candidate within a section."""

    chapter_ref: str
    chapter_title: str
    retrieval_score: float


class HSResearchSectionContext(BaseModel):
    """Section-level context containing candidate chapters."""

    section_id: str
    section_title: str
    section_notes: dict | None
    chapters: list[HSResearchChapterContext]


class HSResearchContext(BaseModel):
    """Legal context provided to the Research Analyst."""

    sections: list[HSResearchSectionContext]


class HSResearchSelectionModel(BaseModel):
    """Minimal chapter selection produced by the Research Analyst."""

    chapter_ref: str


class HSResearchSelectionOutputModel(BaseModel):
    """Minimal structured output produced by the Research Analyst."""

    candidates: list[HSResearchSelectionModel]
