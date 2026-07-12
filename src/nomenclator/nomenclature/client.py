"""HTTP client for the WCO Harmonized System Nomenclature 2022 edition.

This module provides :class:`NomenclatureClient`, which lazily fetches and
parses the HS table of contents, chapter PDFs, section notes, abbreviations,
and general rules from the World Customs Organization website.
"""

from __future__ import annotations

import re

import httpx

from nomenclator.models import (
    HSChapter,
    HSDocumentRef,
    HSGeneralRules,
    HSSection,
    HSSectionNotes,
    HSTree,
)
from nomenclator.nomenclature import parser
from nomenclator.nomenclature.urls import (
    WCO_HS_2022_PDF_BASE_URL,
    WCO_HS_2022_URL,
    chapter_url_from_ref,
    document_url_from_ref,
    normalize_ref_token,
)


class NomenclatureClient:
    """Client for the WCO HS Nomenclature 2022 edition.

    The client is intentionally shaped to be easy to expose as agent tools.
    It loads the HS tree lazily on first use, keeps parsed objects in memory,
    and fetches chapter PDFs on demand with caching.
    """

    def __init__(
        self,
        base_url: str = WCO_HS_2022_URL,
        timeout: int = 30,
        client: httpx.Client | None = None,
        pdf_base_url: str = WCO_HS_2022_PDF_BASE_URL,
    ) -> None:
        """Initialize the nomenclature client.

        Args:
            base_url: Source URL for the HS Nomenclature 2022 table of contents.
            timeout: HTTP timeout in seconds used for page and PDF retrieval.
            client: Optional preconfigured ``httpx.Client``. When omitted, a
                new client is created and closed with :meth:`close`.
            pdf_base_url: Base URL for WCO HS 2022 PDF documents.
        """
        self.base_url = base_url
        self.timeout = timeout
        self._client = client or httpx.Client()
        self._owns_client = client is None
        self._pdf_base_url = pdf_base_url
        self._tree: HSTree | None = None
        self._chapter_cache: dict[str, HSChapter] = {}
        self._section_index: dict[str, HSSection] = {}
        self._chapter_ref_index: dict[str, HSDocumentRef] = {}
        self._chapter_number_index: dict[int, HSDocumentRef] = {}
        self._section_notes_cache: dict[str, HSSectionNotes] = {}
        self._abbreviations_cache: dict[str, str] | None = None
        self._general_rules_cache: HSGeneralRules | None = None

    def close(self) -> None:
        """Close the underlying HTTP client when owned by this instance."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> NomenclatureClient:
        """Enter a context manager and return this client.

        Returns:
            This ``NomenclatureClient`` instance.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context manager and close the client if owned."""
        self.close()

    def get_tree(self) -> HSTree:
        """Return the parsed HS tree.

        The tree is fetched and parsed on first access, then cached in memory.

        Returns:
            The parsed ``HSTree`` instance.
        """
        return self._ensure_tree()

    def get_sections(self) -> list[HSSection]:
        """Return all parsed section nodes.

        Returns:
            A list of ``HSSection`` objects from the nomenclature tree.
        """
        return list(self._ensure_tree().sections)

    def get_node(self, section_id: str) -> HSSection | None:
        """Return a section by its internal identifier.

        Args:
            section_id: Section identifier such as ``section_i``.

        Returns:
            The matching ``HSSection`` when found, otherwise ``None``.
        """
        self._ensure_tree()
        return self._section_index.get(section_id)

    def get_chapter(self, chapter_ref_or_code: str) -> HSChapter:
        """Return a parsed chapter document.

        The input may be any of the following:

        * WCO chapter reference from the tree, for example ``0101-2022E``
        * Short chapter number, for example ``1`` or ``01``
        * Four-digit chapter document code, for example ``0101``

        Parsed chapters are cached in memory after the first fetch.

        Args:
            chapter_ref_or_code: Reference or code identifying the chapter PDF.

        Returns:
            A parsed ``HSChapter`` object with notes, headings, and subheadings.

        Raises:
            ValueError: If the chapter cannot be resolved.
            httpx.HTTPStatusError: If chapter PDF retrieval fails.
        """
        self._ensure_tree()

        normalized_ref = self._normalize_chapter_ref(chapter_ref_or_code)
        if normalized_ref in self._chapter_cache:
            return self._chapter_cache[normalized_ref]

        document = self._chapter_ref_index.get(normalized_ref)
        if document is None:
            raise ValueError(
                f"Unknown chapter reference or code: {chapter_ref_or_code}"
            )

        pdf_url = chapter_url_from_ref(normalized_ref, pdf_base_url=self._pdf_base_url)
        chapter = self._fetch_chapter_pdf(document=document, pdf_url=pdf_url)
        self._chapter_cache[normalized_ref] = chapter
        return chapter

    def get_section_for_chapter(
        self,
        chapter_ref_or_code: str,
    ) -> HSSection:
        """Return the parent section for a chapter reference.

        Args:
            chapter_ref_or_code: WCO chapter reference or chapter code.

        Returns:
            The HS section containing the chapter.

        Raises:
            ValueError: If the chapter cannot be resolved.
        """
        self._ensure_tree()

        normalized_ref = self._normalize_chapter_ref(
            chapter_ref_or_code,
        )

        for section in self._tree.sections:
            for chapter in section.chapters:
                if chapter.ref and normalize_ref_token(chapter.ref) == normalized_ref:
                    return section

        raise ValueError(f"Unable to find section for chapter: {chapter_ref_or_code}")

    def get_section_notes(self, section_id: str) -> HSSectionNotes:
        """Fetch and parse section notes for a section.

        Section notes are fetched from the WCO PDF on first access and cached
        in memory.

        Args:
            section_id: Section identifier such as ``section_i``.

        Returns:
            Parsed section notes.

        Raises:
            ValueError: If the section or its notes reference is not found.
            httpx.HTTPStatusError: If the section notes PDF retrieval fails.
        """
        if section_id in self._section_notes_cache:
            return self._section_notes_cache[section_id]

        ref = self.get_section_notes_ref(section_id)
        if ref is None or not ref.ref:
            raise ValueError(f"No section notes found for {section_id}")

        pdf_url = document_url_from_ref(ref.ref, pdf_base_url=self._pdf_base_url)
        raw_text = self._fetch_pdf_text(pdf_url)
        notes = parser.parse_section_notes_text(raw_text)

        section_notes = HSSectionNotes(
            section_id=section_id,
            title=ref.title,
            document=ref,
            notes=notes,
            raw_text=raw_text,
        )

        self._section_notes_cache[section_id] = section_notes
        return section_notes

    def get_section_notes_ref(self, section_id: str) -> HSDocumentRef | None:
        """Return the section notes document reference for a section.

        Args:
            section_id: Section identifier such as ``section_i``.

        Returns:
            The section notes document reference when present, otherwise
            ``None``.
        """
        section = self.get_node(section_id)
        return section.notes if section else None

    def get_general_rules(self) -> HSGeneralRules:
        """Fetch and parse the General Rules for the Interpretation of the HS.

        The general rules document is fetched on first access and cached in
        memory.

        Returns:
            Parsed general rules.

        Raises:
            ValueError: If the general rules reference is not available.
            httpx.HTTPStatusError: If the general rules PDF retrieval fails.
        """
        if self._general_rules_cache is not None:
            return self._general_rules_cache

        tree = self._ensure_tree()
        ref = tree.general_rules
        if ref is None or not ref.ref:
            raise ValueError("General Rules document reference is not available")

        pdf_url = document_url_from_ref(ref.ref, pdf_base_url=self._pdf_base_url)
        raw_text = self._fetch_pdf_text(pdf_url)
        rules = parser.parse_general_rules(raw_text)

        result = HSGeneralRules(
            title=ref.title,
            document=ref,
            rules=rules,
            raw_text=raw_text,
        )

        self._general_rules_cache = result
        return result

    def get_abbreviations(self) -> dict[str, str]:
        """Return all parsed abbreviations and symbols.

        The abbreviations document is fetched and parsed lazily on first use,
        then cached in memory.

        Returns:
            Mapping from normalized abbreviation term to its definition.

        Raises:
            ValueError: If the abbreviations document reference is missing.
            httpx.HTTPStatusError: If the abbreviations PDF retrieval fails.
        """
        if self._abbreviations_cache is None:
            self._load_abbreviations()

        return dict(self._abbreviations_cache or {})

    def get_abbreviation(self, term: str) -> str | None:
        """Look up a single abbreviation or symbol.

        Args:
            term: Abbreviation or symbol to look up.

        Returns:
            Definition if found, otherwise ``None``.
        """
        if self._abbreviations_cache is None:
            self._load_abbreviations()

        key = parser.normalize_abbreviation_key(term)
        return (
            None
            if self._abbreviations_cache is None
            else self._abbreviations_cache.get(key)
        )

    def _ensure_tree(self) -> HSTree:
        """Load and index the HS tree on first access."""
        if self._tree is None:
            self._tree = self._load_tree()
            self._build_indexes()
        return self._tree

    def _build_indexes(self) -> None:
        """Build internal lookup tables for sections and chapters."""
        if self._tree is None:
            return

        self._section_index = {section.id: section for section in self._tree.sections}
        self._chapter_ref_index = {}
        self._chapter_number_index = {}

        for section in self._tree.sections:
            for chapter in section.chapters:
                if chapter.ref:
                    normalized_ref = normalize_ref_token(chapter.ref)
                    self._chapter_ref_index[normalized_ref] = chapter
                    chapter_number = parser.chapter_number_from_ref(normalized_ref)
                    if chapter_number is not None:
                        self._chapter_number_index[chapter_number] = chapter

    def _load_tree(self) -> HSTree:
        """Fetch and parse the HS table of contents page."""
        response = self._client.get(self.base_url, timeout=self.timeout)
        response.raise_for_status()
        return parser.parse_tree(response.text, source_url=self.base_url)

    def _load_abbreviations(self) -> None:
        """Fetch and parse the abbreviations document into cache."""
        tree = self._ensure_tree()
        if tree.abbreviations is None or not tree.abbreviations.ref:
            raise ValueError("Abbreviations document reference is not available")

        pdf_url = document_url_from_ref(
            tree.abbreviations.ref,
            pdf_base_url=self._pdf_base_url,
        )
        raw_text = self._fetch_pdf_text(pdf_url)
        entries = parser.parse_abbreviations_text(raw_text)

        self._abbreviations_cache = {
            parser.normalize_abbreviation_key(entry.term): entry.definition
            for entry in entries
        }

    def _fetch_chapter_pdf(self, document: HSDocumentRef, pdf_url: str) -> HSChapter:
        """Download and parse a chapter PDF into structured content."""
        raw_text = self._fetch_pdf_text(pdf_url)
        return parser.parse_chapter_text(
            document=document,
            pdf_url=pdf_url,
            raw_text=raw_text,
        )

    def _fetch_pdf_text(self, pdf_url: str) -> str:
        """Download a PDF and return extracted text."""
        response = self._client.get(pdf_url, timeout=self.timeout)
        response.raise_for_status()
        return parser.extract_pdf_text(response.content)

    def _normalize_chapter_ref(self, value: str) -> str:
        """Resolve user input into a canonical chapter reference token."""
        normalized = normalize_ref_token(value)
        if normalized in self._chapter_ref_index:
            return normalized

        digits = re.sub(r"\D", "", normalized)
        if not digits:
            raise ValueError(f"Cannot resolve chapter reference: {value}")

        if len(digits) <= 2:
            chapter_number = int(digits)
            document = self._chapter_number_index.get(chapter_number)
            if document and document.ref:
                return normalize_ref_token(document.ref)

        if len(digits) >= 4:
            candidate_ref = normalize_ref_token(f"{digits[:4]}-2022E")
            if candidate_ref in self._chapter_ref_index:
                return candidate_ref

        raise ValueError(f"Cannot resolve chapter reference: {value}")
