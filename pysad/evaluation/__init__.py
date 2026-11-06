"""
The :mod:`pysad.evaluation` module includes evaluation metrics for anomaly detection on streaming data.
"""

from .metrics import AUPRMetric, AUROCMetric, BaseSKLearnMetric, PrecisionMetric, RecallMetric
from .prequential import evaluate_stream
from .windowed_metric import WindowedMetric

__all__ = [
    "BaseSKLearnMetric",
    "PrecisionMetric",
    "RecallMetric",
    "AUROCMetric",
    "AUPRMetric",
    "WindowedMetric",
    "evaluate_stream",
]
