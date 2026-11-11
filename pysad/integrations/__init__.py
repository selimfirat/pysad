"""Bidirectional integrations between PySAD, PyOD, scikit-learn, and River."""

from typing import Any

from .one_fit_model import OneFitModel
from .predictive_anomaly_detector import PredictiveAnomalyDetector
from .reference_window_model import ReferenceWindowModel
from .sklearn_anomaly_detector import SklearnAnomalyDetector

__all__ = [
    "OneFitModel",
    "PredictiveAnomalyDetector",
    "ReferenceWindowModel",
    "RiverAnomalyDetector",
    "SklearnAnomalyDetector",
]


def __getattr__(name: str) -> Any:
    if name == "RiverAnomalyDetector":
        from .river_anomaly_detector import RiverAnomalyDetector

        return RiverAnomalyDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
