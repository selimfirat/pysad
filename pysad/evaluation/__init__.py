"""
The :mod:`pysad.evaluation` module includes evaluation metrics for anomaly detection on streaming data.
"""

from .metrics import (
    AUPRMetric,
    AUROCMetric,
    BalancedAccuracyMetric,
    BaseSKLearnMetric,
    BrierScoreMetric,
    ExpectedCalibrationErrorMetric,
    F1Metric,
    LogLossMetric,
    MatthewsCorrelationMetric,
    PrecisionMetric,
    RecallMetric,
)
from .prequential import evaluate_postprocessor, evaluate_stream
from .rolling_metric import RollingMetric
from .windowed_metric import WindowedMetric

__all__ = [
    "AUPRMetric",
    "AUROCMetric",
    "BalancedAccuracyMetric",
    "BaseSKLearnMetric",
    "BrierScoreMetric",
    "evaluate_postprocessor",
    "evaluate_stream",
    "ExpectedCalibrationErrorMetric",
    "F1Metric",
    "LogLossMetric",
    "MatthewsCorrelationMetric",
    "PrecisionMetric",
    "RecallMetric",
    "RollingMetric",
    "WindowedMetric",
]
