"""Integration tests for :class:`nomenclator.client.NomenclatureClient`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from nomenclator.nomenclature.client import NomenclatureClient
from nomenclator.nomenclature.tree import HSTree


def test_client_does_not_load_tree_on_init() -> None:
    """Construction should not trigger network access."""
    client = NomenclatureClient()

    try:
        assert client._tree is None
    finally:
        client.close()


def test_close_closes_owned_client() -> None:
    """Closing the client should close an owned HTTP client."""

    http_client = MagicMock()

    client = NomenclatureClient(client=http_client)
    client._owns_client = True

    client.close()

    http_client.close.assert_called_once_with()


def test_close_does_not_close_external_client() -> None:
    """Closing the client should not close an injected HTTP client."""

    http_client = MagicMock()

    client = NomenclatureClient(client=http_client)

    client.close()

    http_client.close.assert_not_called()


@patch.object(NomenclatureClient, "_load_tree")
def test_get_tree_loads_tree_once(
    load_tree,
) -> None:
    """The HS tree should only be loaded once."""

    tree = HSTree(
        source_url="https://example.com",
        sections=[],
        abbreviations={},
        general_rules=None,
    )

    load_tree.return_value = tree

    client = NomenclatureClient()

    try:
        result = client.get_tree()

        assert result is tree
        load_tree.assert_called_once_with()
    finally:
        client.close()


@patch.object(NomenclatureClient, "_load_tree")
def test_get_tree_returns_cached_tree(
    load_tree,
) -> None:
    """Repeated calls should return the cached HS tree."""

    tree = HSTree(
        source_url="https://example.com",
        sections=[],
        abbreviations={},
        general_rules=None,
    )

    load_tree.return_value = tree

    client = NomenclatureClient()

    try:
        tree1 = client.get_tree()
        tree2 = client.get_tree()

        assert tree1 is tree2
        load_tree.assert_called_once_with()
    finally:
        client.close()
