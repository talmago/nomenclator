from dataclasses import asdict, dataclass, field
from typing import Any

from nomenclator.models.tree import HSDocumentRef


@dataclass(slots=True)
class HSNoteClause:
    """Single lettered clause inside a chapter note.

    Args:
        label: Clause label, usually a lower-case letter.
        text: Clause content.
    """

    label: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(slots=True)
class HSNote:
    """Structured chapter note.

    Args:
        number: Ordinal number of the note, if present.
        intro: Introductory note text before any lettered clauses.
        clauses: Parsed lettered clauses.
    """

    number: str | None
    intro: str
    clauses: list[HSNoteClause] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(slots=True)
class HSSectionNotes:
    """Parsed notes for a section.

    Args:
        section_id: Internal section identifier.
        title: Title of the section notes document.
        document: Source document reference.
        notes: Parsed numbered notes.
        raw_text: Full extracted text.
    """

    section_id: str
    title: str
    document: HSDocumentRef
    notes: list[HSNote] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "section_id": self.section_id,
            "title": self.title,
            "document": self.document.to_dict(),
            "notes": [note.to_dict() for note in self.notes],
            "raw_text": self.raw_text,
        }
