"""
The :mod:`pysad.transform.ensemble` module provides ensemblers that combine the scores of several detectors into one score.
"""

from .ensemblers import (
    AverageOfMaximumScoreEnsembler,
    AverageScoreEnsembler,
    MaximumOfAverageScoreEnsembler,
    MaximumScoreEnsembler,
    MedianScoreEnsembler,
)

__all__ = [
    "AverageOfMaximumScoreEnsembler",
    "AverageScoreEnsembler",
    "MaximumOfAverageScoreEnsembler",
    "MaximumScoreEnsembler",
    "MedianScoreEnsembler",
]
