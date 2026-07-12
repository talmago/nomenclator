"""Exceptions raised by the HS classification pipeline."""


class HSClassificationError(Exception):
    """Base exception for HS classification failures."""


class HSInitializationError(HSClassificationError):
    """Raised when the classification agent cannot initialize required resources."""


class HSClassificationPipelineError(HSClassificationError):
    """Raised when the classification pipeline cannot complete."""


class HSNoCandidatesFoundError(HSClassificationPipelineError):
    """Raised when retrieval produces no HS chapter candidates."""
