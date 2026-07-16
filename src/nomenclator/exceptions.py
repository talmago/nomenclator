"""Exceptions raised by the HS classification pipeline."""


class HSClassificationError(Exception):
    """Base exception for HS classification failures."""


class HSInitializationError(HSClassificationError):
    """Raised when the classification agent cannot initialize required resources."""


class HSClassificationPipelineError(HSClassificationError):
    """Raised when the classification pipeline cannot complete."""


class HSNoCandidatesFoundError(HSClassificationPipelineError):
    """Raised when retrieval produces no HS chapter candidates."""


class HSProductAnalysisError(HSClassificationPipelineError):
    """Raised when product analysis fails."""


class HSResearchAnalysisError(HSClassificationPipelineError):
    """Raised when HS research analysis fails."""


class HSClassificationAnalysisError(HSClassificationPipelineError):
    """Raised when final classification analysis fails."""
