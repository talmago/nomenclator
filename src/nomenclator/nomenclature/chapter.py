from dataclasses import dataclass, field
from typing import Any

from nomenclator.nomenclature.notes import HSNote
from nomenclator.nomenclature.tree import HSDocumentRef, HSHeading


@dataclass(slots=True)
class HSChapter:
    """Structured HS chapter document.

    Args:
        chapter_number: Numeric chapter number, for example ``1``.
        title: Chapter title.
        document: Source document reference for the chapter PDF.
        notes: Parsed chapter notes.
        headings: Parsed headings and subheadings.
        raw_text: Full extracted PDF text.
    """

    chapter_number: int
    title: str
    document: HSDocumentRef
    notes: list[HSNote] = field(default_factory=list)
    headings: list[HSHeading] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "chapter_number": self.chapter_number,
            "title": self.title,
            "document": self.document.to_dict(),
            "notes": [note.to_dict() for note in self.notes],
            "headings": [heading.to_dict() for heading in self.headings],
            "raw_text": self.raw_text,
        }
