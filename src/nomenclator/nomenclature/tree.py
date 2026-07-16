from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class HSDocumentRef:
    """Reference to a WCO HS document.

    Args:
        title: Human-readable title.
        ref: WCO reference token from the table of contents, such as
            ``0101-2022E`` or ``0001-2202E GIR``.
        url: Absolute URL when known.
    """

    title: str
    ref: str | None = None
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(slots=True)
class HSSubheading:
    """Structured HS subheading.

    Args:
        code: Six-digit HS code rendered with a decimal point, for example
            ``0101.21``.
        description: Human-readable text for the subheading.
        group_path: Active grouping labels that applied when the subheading was
            parsed, ordered from outermost to innermost.
    """

    code: str
    description: str
    group_path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(slots=True)
class HSHeading:
    """Structured HS heading inside a chapter.

    Args:
        code: Four-digit heading code rendered with a decimal point, for example
            ``01.01``.
        description: Human-readable text for the heading.
        subheadings: Parsed subheadings under this heading.
    """

    code: str
    description: str
    subheadings: list[HSSubheading] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(slots=True)
class HSSection:
    """HS section node from the nomenclature tree.

    Args:
        id: Stable internal identifier such as ``section_i``.
        label: Display label such as ``SECTION I``.
        title: Section title.
        notes: Reference to the section notes PDF.
        chapters: Chapter document references listed under the section.
    """

    id: str
    label: str
    title: str
    notes: HSDocumentRef | None = None
    chapters: list[HSDocumentRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "id": self.id,
            "label": self.label,
            "title": self.title,
            "notes": self.notes.to_dict() if self.notes else None,
            "chapters": [chapter.to_dict() for chapter in self.chapters],
        }


@dataclass(slots=True)
class HSTree:
    """Top-level HS nomenclature tree.

    Args:
        source_url: Source page used to build the tree.
        introduction: Optional introduction document reference.
        abbreviations: Optional abbreviations document reference.
        general_rules: Optional General Rules document reference.
        sections: Parsed section nodes.
    """

    source_url: str
    introduction: HSDocumentRef | None = None
    abbreviations: HSDocumentRef | None = None
    general_rules: HSDocumentRef | None = None
    sections: list[HSSection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "source_url": self.source_url,
            "introduction": self.introduction.to_dict() if self.introduction else None,
            "abbreviations": self.abbreviations.to_dict()
            if self.abbreviations
            else None,
            "general_rules": self.general_rules.to_dict()
            if self.general_rules
            else None,
            "sections": [section.to_dict() for section in self.sections],
        }
