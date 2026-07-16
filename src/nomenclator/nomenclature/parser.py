"""Parsing utilities for WCO Harmonized System nomenclature documents.

This module converts raw HTML and PDF text from the WCO HS 2022 edition into
structured :mod:`nomenclator.models` objects such as trees, chapters, notes,
abbreviations, and general rules.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from nomenclator.models import (
    HSAbbreviation,
    HSChapter,
    HSDocumentRef,
    HSGeneralRule,
    HSHeading,
    HSNote,
    HSNoteClause,
    HSSection,
    HSSubheading,
    HSTree,
)
from nomenclator.nomenclature.urls import normalize_ref_token


def parse_tree(html: str, source_url: str) -> HSTree:
    """Parse the HS table of contents page into structured objects.

    Args:
        html: Raw HTML returned by the WCO nomenclature page.
        source_url: URL from which the HTML was fetched.

    Returns:
        Parsed ``HSTree`` containing sections, chapters, and document refs.
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n")
    lines = [_clean_line(line) for line in text.splitlines()]
    tokens = [line for line in lines if line]

    tree = HSTree(source_url=source_url)
    current_section: HSSection | None = None
    i = 0

    while i < len(tokens):
        line = tokens[i]

        if line == "Introduction":
            ref = _next_ref_token(tokens, i + 1)
            tree.introduction = HSDocumentRef(title="Introduction", ref=ref)
            i += 1
            continue

        if line.startswith("Abbreviations and symbols"):
            ref = _next_ref_token(tokens, i + 1)
            tree.abbreviations = HSDocumentRef(
                title="Abbreviations and symbols", ref=ref
            )
            i += 1
            continue

        if line.startswith(
            "General Rules for the interpretation of the Harmonized System"
        ):
            ref = _next_ref_token(tokens, i + 1)
            tree.general_rules = HSDocumentRef(
                title="General Rules for the interpretation of the Harmonized System",
                ref=ref,
            )
            i += 1
            continue

        if _is_section_label(line):
            label = line
            title = _safe_token(tokens, i + 1)
            current_section = HSSection(
                id=_section_id_from_label(label),
                label=label,
                title=title,
            )
            tree.sections.append(current_section)
            i += 2
            continue

        if current_section and line.startswith("Section Note"):
            ref = _next_ref_token(tokens, i + 1)
            current_section.notes = HSDocumentRef(title=line, ref=ref)
            i += 1
            continue

        if current_section and _is_chapter_number(line):
            chapter_number = int(line)
            title = _safe_token(tokens, i + 1)
            ref = _next_ref_token(tokens, i + 2)
            current_section.chapters.append(
                HSDocumentRef(
                    title=title,
                    ref=ref or _chapter_ref_from_number(chapter_number),
                )
            )
            i += 2
            continue

        i += 1

    return tree


def parse_chapter_text(
    document: HSDocumentRef, pdf_url: str, raw_text: str
) -> HSChapter:
    """Parse raw chapter text into a structured chapter object.

    Args:
        document: Chapter document reference from the nomenclature tree.
        pdf_url: Absolute URL of the source chapter PDF.
        raw_text: Text extracted from the chapter PDF.

    Returns:
        Structured ``HSChapter`` with notes, headings, and subheadings.

    Raises:
        ValueError: If the chapter number cannot be determined from the text
            or document reference.
    """
    lines = [clean_pdf_line(line) for line in raw_text.splitlines()]
    lines = [line for line in lines if line]

    chapter_number: int | None = None
    title = document.title
    notes: list[HSNote] = []
    headings: list[HSHeading] = []

    current_heading: HSHeading | None = None
    current_subheading: HSSubheading | None = None
    active_groups: dict[int, str] = {}
    in_notes = False
    current_note: HSNote | None = None
    current_clause: HSNoteClause | None = None

    for line in lines:
        if _is_table_noise(line):
            continue

        chapter_header_match = re.match(r"^Chapter\s+(\d+)$", line, flags=re.IGNORECASE)
        if chapter_header_match:
            chapter_number = int(chapter_header_match.group(1))
            continue

        if (
            chapter_number is not None
            and title == document.title
            and line.lower() != document.title.lower()
            and not notes
            and not headings
            and not line.lower().startswith("note")
        ):
            title = line
            continue

        if line.startswith("Note"):
            in_notes = True
            current_note = None
            current_clause = None
            continue

        if in_notes:
            if _is_separator(line):
                if current_clause and current_note:
                    current_note.clauses.append(current_clause)
                    current_clause = None
                if current_note:
                    notes.append(current_note)
                    current_note = None
                in_notes = False
                continue

            note_header_match = re.match(r"^(\d+)\.\-\s*(.*)$", line)
            if note_header_match:
                if current_clause and current_note:
                    current_note.clauses.append(current_clause)
                    current_clause = None
                if current_note:
                    notes.append(current_note)
                current_note = HSNote(
                    number=note_header_match.group(1),
                    intro=note_header_match.group(2).strip(),
                )
                continue

            clause_match = re.match(r"^\(([a-z])\)\s*(.*)$", line)
            if clause_match:
                if current_note is None:
                    current_note = HSNote(number=None, intro="")
                if current_clause:
                    current_note.clauses.append(current_clause)
                current_clause = HSNoteClause(
                    label=clause_match.group(1),
                    text=clause_match.group(2).strip(),
                )
                continue

            if current_clause is not None:
                current_clause.text = f"{current_clause.text} {line}".strip()
            elif current_note is not None:
                current_note.intro = f"{current_note.intro} {line}".strip()
            else:
                current_note = HSNote(number=None, intro=line)
            continue

        heading_match = re.match(r"^(\d{2}\.\d{2})\s+(.+)$", line)
        if heading_match:
            if current_heading:
                headings.append(current_heading)
            current_heading = HSHeading(
                code=heading_match.group(1),
                description=heading_match.group(2).strip(),
            )
            current_subheading = None
            active_groups.clear()
            continue

        subheading_match = re.match(r"^(\d{4}\.\d{2})\s*(.*)$", line)
        if subheading_match and current_heading:
            raw_desc = subheading_match.group(2).strip()

            if raw_desc.startswith("--"):
                description = raw_desc.lstrip("-").strip()
                group_path = _group_path(active_groups)
            elif raw_desc.startswith("-"):
                description = raw_desc.lstrip("-").strip()
                group_path = []
            else:
                description = raw_desc
                group_path = _group_path(active_groups)

            current_subheading = HSSubheading(
                code=subheading_match.group(1),
                description=description,
                group_path=group_path,
            )
            current_heading.subheadings.append(current_subheading)
            continue

        group_match = re.match(r"^(\-+)\s*(.+?)\s*:?$", line)
        if group_match and current_heading:
            level = len(group_match.group(1))
            label = group_match.group(2).strip()
            active_groups[level] = label
            active_groups = {
                depth: value for depth, value in active_groups.items() if depth <= level
            }
            current_subheading = None
            continue

        if current_subheading is not None:
            current_subheading.description = (
                f"{current_subheading.description} {line}".strip()
            )
            continue

        if current_heading is not None:
            current_heading.description = (
                f"{current_heading.description} {line}".strip()
            )
            continue

    if current_clause and current_note:
        current_note.clauses.append(current_clause)
    if current_note:
        notes.append(current_note)
    if current_heading:
        headings.append(current_heading)

    if chapter_number is None:
        inferred = chapter_number_from_ref(normalize_ref_token(document.ref or ""))
        if inferred is None:
            raise ValueError(
                f"Could not determine chapter number for document ref: {document.ref}"
            )
        chapter_number = inferred

    document.url = pdf_url
    return HSChapter(
        chapter_number=chapter_number,
        title=title,
        document=document,
        notes=notes,
        headings=headings,
        raw_text=raw_text,
    )


def parse_section_notes_text(raw_text: str) -> list[HSNote]:
    """Parse section notes from extracted PDF text.

    Args:
        raw_text: Text extracted from a section notes PDF.

    Returns:
        Parsed numbered notes from the notes section of the document.

    Raises:
        ValueError: If the notes section cannot be located in the text.
    """
    lines = [clean_pdf_line(line) for line in raw_text.splitlines()]
    lines = [line for line in lines if line]

    start_idx = None
    for i, line in enumerate(lines):
        if line.lower() == "notes.":
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError("Could not locate notes section")

    return parse_notes_block(lines[start_idx:])


def parse_notes_block(lines: list[str]) -> list[HSNote]:
    """Parse a block of cleaned lines containing numbered notes.

    Args:
        lines: Cleaned lines representing a notes section.

    Returns:
        List of parsed ``HSNote`` objects.
    """
    notes: list[HSNote] = []

    current_note: HSNote | None = None
    current_clause: HSNoteClause | None = None

    for line in lines:
        if _is_separator(line):
            if current_clause and current_note:
                current_note.clauses.append(current_clause)
                current_clause = None
            if current_note:
                notes.append(current_note)
                current_note = None
            break

        note_header_match = re.match(r"^(\d+)\.\-\s*(.*)$", line)
        if note_header_match:
            if current_clause and current_note:
                current_note.clauses.append(current_clause)
                current_clause = None
            if current_note:
                notes.append(current_note)

            current_note = HSNote(
                number=note_header_match.group(1),
                intro=note_header_match.group(2).strip(),
            )
            continue

        clause_match = re.match(r"^\(([a-z])\)\s*(.*)$", line)
        if clause_match:
            if current_note is None:
                current_note = HSNote(number=None, intro="")

            if current_clause:
                current_note.clauses.append(current_clause)

            current_clause = HSNoteClause(
                label=clause_match.group(1),
                text=clause_match.group(2).strip(),
            )
            continue

        if current_clause is not None:
            current_clause.text = f"{current_clause.text} {line}".strip()
        elif current_note is not None:
            current_note.intro = f"{current_note.intro} {line}".strip()
        else:
            current_note = HSNote(number=None, intro=line)

    if current_clause and current_note:
        current_note.clauses.append(current_clause)
    if current_note:
        notes.append(current_note)

    return notes


def parse_general_rules(raw_text: str) -> list[HSGeneralRule]:
    """Parse General Rules for the Interpretation of the HS.

    Args:
        raw_text: Text extracted from the general rules PDF.

    Returns:
        List of parsed ``HSGeneralRule`` entries.
    """
    lines = [clean_pdf_line(line) for line in raw_text.splitlines()]
    lines = [line for line in lines if line]

    rules: list[HSGeneralRule] = []
    current_rule: HSGeneralRule | None = None

    for line in lines:
        match = re.match(r"^(\d+)\.\s*(.*)", line)
        if match:
            if current_rule:
                rules.append(current_rule)

            current_rule = HSGeneralRule(
                rule=match.group(1),
                text=match.group(2).strip(),
            )
            continue

        sub_match = re.match(r"^\(([a-z])\)\s*(.*)", line)
        if sub_match and current_rule:
            label = f"{current_rule.rule}({sub_match.group(1)})"
            rules.append(
                HSGeneralRule(
                    rule=label,
                    text=sub_match.group(2).strip(),
                )
            )
            continue

        if current_rule:
            current_rule.text += " " + line

    if current_rule:
        rules.append(current_rule)

    return rules


def parse_abbreviations_text(raw_text: str) -> list[HSAbbreviation]:
    """Parse the abbreviations and symbols document.

    Args:
        raw_text: Text extracted from the abbreviations PDF.

    Returns:
        List of parsed ``HSAbbreviation`` entries.
    """
    entries: list[HSAbbreviation] = []

    lines = [clean_pdf_line(line) for line in raw_text.splitlines()]
    lines = [line for line in lines if line]

    for line in lines:
        if line.upper() == "ABBREVIATIONS AND SYMBOLS":
            continue

        if line.lower().startswith("examples"):
            break

        if _is_separator(line):
            continue

        parsed = _split_abbreviation_line(line)
        if parsed is None:
            continue

        term, definition = parsed
        entries.append(
            HSAbbreviation(
                term=term,
                definition=definition,
            )
        )

    return entries


def normalize_abbreviation_key(term: str) -> str:
    """Normalize an abbreviation lookup key.

    Args:
        term: Raw abbreviation or symbol.

    Returns:
        Upper-case, whitespace-normalized lookup key.
    """
    return re.sub(r"\s+", " ", term.strip().upper())


def chapter_number_from_ref(ref: str) -> int | None:
    """Extract the chapter number from a WCO chapter reference.

    Args:
        ref: Canonical chapter reference token such as ``0101-2022E``.

    Returns:
        Chapter number when it can be inferred, otherwise ``None``.
    """
    match = re.match(r"^(\d{2})(\d{2})-2022E", ref)
    if not match:
        return None
    return int(match.group(2))


def clean_pdf_line(line: str) -> str:
    """Normalize a line extracted from PDF text.

    Args:
        line: Raw line text from a PDF extractor.

    Returns:
        Normalized line with consistent whitespace and dash characters.
    """
    line = line.replace("\u2013", "-").replace("\u2014", "-")
    line = line.replace("\xa0", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def _chapter_ref_from_number(chapter_number: int) -> str:
    return f"{chapter_number:02d}{chapter_number:02d}-2022E"


def _split_abbreviation_line(line: str) -> tuple[str, str] | None:
    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        return None

    term, definition = parts
    if len(term) <= 10:
        return term.strip(), definition.strip()

    return None


def _group_path(active_groups: dict[int, str]) -> list[str]:
    return [label for _, label in sorted(active_groups.items())]


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.replace("\xa0", " ")).strip()


def _is_table_noise(line: str) -> bool:
    normalized = line.lower().replace(" ", "")
    return normalized in {"headingh.s.", "code", "headingh.s.code"}


def _is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"[_\-]{5,}", line))


def _is_section_label(line: str) -> bool:
    return bool(re.match(r"^SECTION\s+[IVXLC]+$", line))


def _section_id_from_label(label: str) -> str:
    roman = label.rsplit(maxsplit=1)[-1].lower()
    return f"section_{roman}"


def _is_chapter_number(line: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}", line))


def _safe_token(tokens: list[str], index: int) -> str:
    return tokens[index] if 0 <= index < len(tokens) else ""


def _next_ref_token(tokens: list[str], start_index: int) -> str | None:
    for index in range(start_index, min(start_index + 6, len(tokens))):
        token = tokens[index]
        if re.fullmatch(r"[A-Z0-9]+-(?:2022E|2202E)(?:\s+[A-Z]+)?", token):
            return token.strip()

    return None
