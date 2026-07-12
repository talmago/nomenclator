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
    power_source: str | None = None
    is_part: bool | None = None


class ProductFactsModel(BaseModel):
    """Structured product facts extracted from free-text description."""

    raw_description: str
    normalized_description: str
    product_category: str | None = None
    main_attributes: MainAttributesModel
    keywords: list[str]
    hints: list[UserHSHintModel]
