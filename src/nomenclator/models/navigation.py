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


class HSResearchCandidateModel(BaseModel):
    """HS chapter pathway identified by the Research Analyst."""

    section_id: str
    section_title: str
    chapter_ref: str
    chapter_title: str

    score: float
    reason: list[str]


class HSResearchOutputModel(BaseModel):
    """Ranked HS chapter pathways produced by the Research Analyst."""

    candidates: list[HSResearchCandidateModel]
