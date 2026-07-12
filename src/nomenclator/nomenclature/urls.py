from __future__ import annotations

import re

WCO_HS_2022_URL = (
    "https://www.wcoomd.org/en/topics/nomenclature/"
    "instrument-and-tools/hs-nomenclature-2022-edition/"
    "hs-nomenclature-2022-edition.aspx"
)

WCO_HS_2022_PDF_BASE_URL = (
    "https://www.wcoomd.org/-/media/wco/public/global/pdf/"
    "topics/nomenclature/instruments-and-tools/hs-nomenclature-2022/2022"
)


def normalize_ref_token(ref: str) -> str:
    """Normalize a WCO reference token."""
    return re.sub(r"\s+", " ", ref.strip().upper())


def chapter_url_from_ref(
    ref: str,
    pdf_base_url: str = WCO_HS_2022_PDF_BASE_URL,
) -> str:
    """Build the PDF URL for a chapter reference."""
    match = re.match(r"^(\d{4})-2022E", ref)
    if not match:
        raise ValueError(f"Cannot build chapter URL from ref: {ref}")
    return f"{pdf_base_url}/{match.group(1)}_2022e.pdf?la=en"


def document_url_from_ref(
    ref: str,
    pdf_base_url: str = WCO_HS_2022_PDF_BASE_URL,
) -> str:
    """Build a PDF URL from a WCO reference token."""
    normalized = normalize_ref_token(ref)

    if re.match(r"^\d{4}-2022E(?: .+)?$", normalized):
        return chapter_url_from_ref(normalized, pdf_base_url=pdf_base_url)

    special_map = {
        "ABBREV-2022E": "ABBREV_2022E.pdf?la=en",
        "INTRODUCTION-2022E": "Introduction_2022E.pdf?la=en",
        "0001-2202E GIR": "0001_2022e-gir.pdf?la=en",
    }

    if normalized in special_map:
        return f"{pdf_base_url}/{special_map[normalized]}"

    match = re.match(r"^([A-Z0-9]+)-2022E(?: .+)?$", normalized)
    if match:
        return f"{pdf_base_url}/{match.group(1)}_2022e.pdf?la=en"

    raise ValueError(f"Cannot build document URL from ref: {ref}")
