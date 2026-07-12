"""Integration tests for :class:`nomenclator.client.NomenclatureClient`."""

from __future__ import annotations

import pytest

from nomenclator.nomenclature.client import NomenclatureClient


@pytest.mark.integration
def test_wco_hs_nomenclature_2022_client_integration() -> None:
    """Download and parse the official WCO HS Nomenclature 2022 edition."""
    client = NomenclatureClient()
    tree = client.get_tree()

    assert tree.source_url == client.base_url
    assert len(tree.sections) == 21
    assert tree.abbreviations is not None
    assert tree.general_rules is not None

    section_i = tree.sections[0]
    assert section_i.id == "section_i"
    assert section_i.label == "SECTION I"
    assert section_i.title == "LIVE ANIMALS; ANIMAL PRODUCTS"
    assert section_i.chapters[0].ref == "0101-2022E"
    assert section_i.chapters[0].title == "Live animals."

    abbreviations = client.get_abbreviations()

    assert len(abbreviations) >= 40
    assert abbreviations["AC"] == "alternating current"
    assert abbreviations["CM"] == "centimetre(s)"

    section_notes = client.get_section_notes("section_i")

    assert section_notes.section_id == "section_i"
    assert len(section_notes.notes) >= 2

    first_note = section_notes.notes[0]
    assert first_note.number == "1"
    assert first_note.intro
    assert "genus" in first_note.intro.lower()

    second_note = section_notes.notes[1]
    assert second_note.number == "2"
    assert second_note.intro

    chapter = client.get_chapter("01")

    assert chapter.chapter_number == 1
    assert chapter.title == "Live animals"
    assert chapter.notes
    assert chapter.headings

    first_note = chapter.notes[0]
    assert first_note.number == "1"
    assert first_note.clauses
    assert first_note.clauses[0].label == "a"

    heading_codes = [heading.code for heading in chapter.headings[:3]]
    assert heading_codes == ["01.01", "01.02", "01.03"]

    horses = chapter.headings[0].subheadings[:2]
    assert [sub.code for sub in horses] == ["0101.21", "0101.29"]
    assert horses[0].group_path == ["Horses"]
    assert horses[0].description == "Pure-bred breeding animals"
