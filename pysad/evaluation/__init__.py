"""
The :mod:`pysad.evaluation` module includes evaluation metrics for anomaly detection on streaming data.
"""

from .f1_metric import F1Metric
from .metrics import (
    AUPRMetric,
    AUROCMetric,
    BaseSKLearnMetric,
    MatthewsCorrelationMetric,
    PrecisionMetric,
    RecallMetric,
)
from .prequential import evaluate_stream
from .windowed_metric import WindowedMetric

__all__ = [
    "AUPRMetric",
    "AUROCMetric",
    "BaseSKLearnMetric",
    "F1Metric",
    "MatthewsCorrelationMetric",
    "PrecisionMetric",
    "RecallMetric",
    "WindowedMetric",
    "evaluate_stream",
]
