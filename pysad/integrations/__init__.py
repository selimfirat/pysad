"""Bidirectional integrations between PySAD, PyOD, scikit-learn, and River."""

from typing import Any

from .one_fit_model import OneFitModel
from .predictive_anomaly_detector import PredictiveAnomalyDetector
from .pysad_to_sklearn_adapter import PySADToScikitLearnAdapter
from .reference_window_model import ReferenceWindowModel

__all__ = [
    "OneFitModel",
    "PredictiveAnomalyDetector",
    "PySADToRiverAdapter",
    "PySADToScikitLearnAdapter",
    "ReferenceWindowModel",
    "RiverDriftDetector",
]


def __getattr__(name: str) -> Any:
    if name == "PySADToRiverAdapter":
        from .pysad_to_river_adapter import PySADToRiverAdapter

        return PySADToRiverAdapter
    if name == "RiverDriftDetector":
        from .river_drift_detector import RiverDriftDetector

        return RiverDriftDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
