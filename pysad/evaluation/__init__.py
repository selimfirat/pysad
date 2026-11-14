"""
The :mod:`pysad.evaluation` module includes evaluation metrics for anomaly detection on streaming data.
"""

from .metrics import (
    AUPRMetric,
    AUROCMetric,
    BalancedAccuracyMetric,
    BaseSKLearnMetric,
    F1Metric,
    LogLossMetric,
    MatthewsCorrelationMetric,
    PrecisionMetric,
    RecallMetric,
)
from .prequential import evaluate_stream
from .rolling_metric import RollingMetric
from .windowed_metric import WindowedMetric

__all__ = [
    "AUPRMetric",
    "AUROCMetric",
    "BalancedAccuracyMetric",
    "BaseSKLearnMetric",
    "evaluate_stream",
    "F1Metric",
    "LogLossMetric",
    "MatthewsCorrelationMetric",
    "PrecisionMetric",
    "RecallMetric",
    "RollingMetric",
    "WindowedMetric",
]
