from typing import Literal

from pydantic import BaseModel


class UserHSHintModel(BaseModel):
    """User-provided HS code hint."""

    code: str
    source: Literal["user"] = "user"
    verified: bool = False


class MainAttributesModel(BaseModel):
    """Core extracted attributes of the product."""

    product_type: str | None = None
    function: str | None = None
    material: str | None = None
    is_part: bool | None = None


class ProductFactsModel(BaseModel):
    """Structured product facts extracted from a free-text description."""

    raw_description: str
    normalized_description: str
    product_category: str | None = None
    main_attributes: MainAttributesModel
    keywords: list[str]
    hints: list[UserHSHintModel]

    def retrieval_query(self) -> str:
        """Build a query for HS nomenclature retrieval."""

        terms = [
            self.normalized_description,
            self.product_category,
            self.main_attributes.product_type,
            *self.keywords,
        ]

        sections = [
            term.strip() for term in dict.fromkeys(terms) if term and term.strip()
        ]

        if self.hints:
            hints = ", ".join(
                dict.fromkeys(
                    hint.code.strip() for hint in self.hints if hint.code.strip()
                )
            )
            if hints:
                sections.append(f"HS hints: {hints}")

        return "\n".join(sections)
