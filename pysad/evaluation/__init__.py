"""
The :mod:`pysad.evaluation` module includes evaluation metrics for anomaly detection on streaming data.
"""

from .f1_metric import F1Metric
from .metrics import AUPRMetric, AUROCMetric, BaseSKLearnMetric, PrecisionMetric, RecallMetric
from .prequential import evaluate_stream
from .rolling_metric import RollingMetric
from .windowed_metric import WindowedMetric

__all__ = [
    "AUPRMetric",
    "AUROCMetric",
    "BaseSKLearnMetric",
    "F1Metric",
    "PrecisionMetric",
    "RecallMetric",
    "RollingMetric",
    "WindowedMetric",
    "evaluate_stream",
]
