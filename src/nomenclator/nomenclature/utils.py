from io import BytesIO

from pypdf import PdfReader


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract plain text from a PDF document.

    Args:
        pdf_bytes: Raw PDF file bytes.

    Returns:
        Text extracted from all pages, joined with newline characters.
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)
