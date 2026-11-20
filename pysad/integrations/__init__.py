"""Bidirectional integrations between PySAD, PyOD, scikit-learn, and River."""

from typing import Any

from .one_fit_model import OneFitModel
from .predictive_anomaly_detector import PredictiveAnomalyDetector
from .reference_window_model import ReferenceWindowModel
from .river_drift_detector import RiverDriftDetector
from .sklearn_anomaly_detector import SklearnAnomalyDetector

__all__ = [
    "OneFitModel",
    "PredictiveAnomalyDetector",
    "ReferenceWindowModel",
    "RiverAnomalyDetector",
    "RiverDriftDetector",
    "SklearnAnomalyDetector",
]


def __getattr__(name: str) -> Any:
    if name == "RiverAnomalyDetector":
        from .river_anomaly_detector import RiverAnomalyDetector

        return RiverAnomalyDetector
    if name == "RiverDriftDetector":
        from .river_drift_detector import RiverDriftDetector

        return RiverDriftDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
