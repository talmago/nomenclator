"""Embedding model abstraction for semantic retrieval.

This module provides a common interface for encoding text into dense vector
embeddings independently of the underlying embedding library.

Embedding providers are exposed through lightweight adapter classes
implementing the :class:`EmbeddingModel` protocol. Models are loaded lazily
and cached for reuse, allowing retrieval indexes to share embedding models
without repeated initialization.

Currently supported providers are:

* Model2Vec static embedding models.
* FastEmbed ONNX-based embedding models.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import Protocol

from huggingface_hub.utils.tqdm import disable_progress_bars
from model2vec import StaticModel
import numpy as np
import numpy.typing as npt

EmbeddingArray = npt.NDArray[np.float32]


class EmbeddingModel(Protocol):
    """Protocol implemented by dense embedding models.

    Implementations encode one or more text documents into dense vector
    embeddings suitable for semantic similarity search.
    """

    def encode(
        self,
        texts: Sequence[str],
    ) -> EmbeddingArray:
        """Encode text into dense embedding vectors.

        Args:
            texts: Documents to encode.

        Returns:
            Two-dimensional embedding array where each row corresponds to one
            input document.
        """
        ...


class Model2VecEmbeddingModel:
    """Embedding model backed by a Model2Vec static model."""

    def __init__(
        self,
        model: StaticModel,
    ) -> None:
        """Initialize the embedding model.

        Args:
            model: Loaded Model2Vec static model.
        """
        self._model = model

    def encode(
        self,
        texts: Sequence[str],
    ) -> EmbeddingArray:
        """Encode text into dense embedding vectors.

        Args:
            texts: Documents to encode.

        Returns:
            Two-dimensional embedding array.
        """
        embeddings = self._model.encode(
            list(texts),
            use_multiprocessing=False,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )


class FastEmbedEmbeddingModel:
    """Embedding model backed by FastEmbed.

    FastEmbed provides lightweight ONNX-based embedding models suitable for
    production deployments without requiring a full PyTorch runtime.
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:
        """Initialize the embedding model.

        Args:
            model_name: FastEmbed model identifier.

        Raises:
            ImportError: If the ``fastembed`` package is not installed.
        """
        try:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "FastEmbed support requires the optional 'fastembed' package. "
                "Install it with the appropriate extra dependencies."
            ) from exc

        self._model = TextEmbedding(model_name=model_name)

    def encode(
        self,
        texts: Sequence[str],
    ) -> EmbeddingArray:
        """Encode text into dense embedding vectors.

        Args:
            texts: Text values to encode.

        Returns:
            Two-dimensional array with one embedding per input text.
        """
        values = list(texts)

        if not values:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = list(self._model.embed(values))

        result = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if result.ndim != 2:
            raise ValueError(
                "Embedding model must return a two-dimensional array, "
                f"received shape {result.shape}"
            )

        if result.shape[0] != len(values):
            raise ValueError(
                "Embedding count does not match input count: "
                f"{result.shape[0]} embeddings for {len(values)} texts"
            )

        return result


@cache
def _load_cached_model2vec(
    model_name: str,
) -> StaticModel:
    """Load and cache a Model2Vec embedding model.

    Models are cached after their first initialization to avoid repeated
    downloads and startup overhead.

    Args:
        model_name: Model2Vec model identifier.

    Returns:
        Loaded Model2Vec model.
    """
    disable_progress_bars()

    try:
        return StaticModel.from_pretrained(
            model_name,
            force_download=False,
        )
    finally:
        disable_progress_bars()


def load_embedding_model(
    model_name: str,
) -> EmbeddingModel:
    """Load an embedding model.

    The provider is selected from the model identifier.

    Currently, Model2Vec models are detected by the ``minishlab/`` prefix.
    All other model identifiers are delegated to FastEmbed.

    Args:
        model_name: Embedding model identifier.

    Returns:
        Loaded embedding model.
    """

    if model_name.startswith("minishlab/"):
        return Model2VecEmbeddingModel(
            _load_cached_model2vec(model_name),
        )

    return FastEmbedEmbeddingModel(model_name)
