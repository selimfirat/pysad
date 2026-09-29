"""
The :mod:`pysad.transform.ensemble` module consist of ensemblers to combine scores from multiple anomaly detectors.
"""

from .ensemblers import (
    AverageOfMaximumScoreEnsembler,
    AverageScoreEnsembler,
    MaximumOfAverageScoreEnsembler,
    MaximumScoreEnsembler,
    MedianScoreEnsembler,
)

__all__ = [
    "MaximumScoreEnsembler",
    "MedianScoreEnsembler",
    "AverageScoreEnsembler",
    "MaximumOfAverageScoreEnsembler",
    "AverageOfMaximumScoreEnsembler",
]
