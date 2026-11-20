"""
The :mod:`pysad.explainability` module provides explainability methods for streaming
anomaly detectors, including TreeSHAP for tree ensembles.
"""

from .treeshap import treeshap_ensemble, treeshap_ensemble_weighted

__all__ = [
    "treeshap_ensemble",
    "treeshap_ensemble_weighted",
]
