from dataclasses import asdict, dataclass, field
from typing import Any

from nomenclator.nomenclature.tree import HSDocumentRef


@dataclass(slots=True)
class HSGeneralRule:
    """Single General Rule entry.

    Args:
        rule: Rule identifier (e.g. "1", "2(a)", "3(b)").
        text: Full rule text.
    """

    rule: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable representation."""
        return asdict(self)


@dataclass(slots=True)
class HSGeneralRules:
    """General Rules for the Interpretation of the Harmonized System.

    Args:
        title: Document title.
        document: Source document reference.
        rules: Parsed list of rules.
        raw_text: Full extracted text.
    """

    title: str
    document: HSDocumentRef
    rules: list[HSGeneralRule] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable representation."""
        return {
            "title": self.title,
            "document": self.document.to_dict(),
            "rules": [rule.to_dict() for rule in self.rules],
            "raw_text": self.raw_text,
        }


@dataclass(slots=True)
class HSAbbreviation:
    """Parsed abbreviation entry.

    Args:
        term: Abbreviation or symbol.
        definition: Human-readable definition.
    """

    term: str
    definition: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)
