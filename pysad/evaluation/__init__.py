"""
The :mod:`pysad.evaluation` module includes evaluation metrics for anomaly detection on streaming data.
"""

from .balanced_accuracy import BalancedAccuracyMetric
from .metrics import AUPRMetric, AUROCMetric, BaseSKLearnMetric, PrecisionMetric, RecallMetric
from .prequential import evaluate_stream
from .windowed_metric import WindowedMetric

__all__ = [
    "AUPRMetric",
    "AUROCMetric",
    "BalancedAccuracyMetric",
    "BaseSKLearnMetric",
    "PrecisionMetric",
    "RecallMetric",
    "WindowedMetric",
    "evaluate_stream",
]
