"""Adapters for using PySAD components in other libraries."""

from typing import Any

from .sklearn_anomaly_detector import SklearnAnomalyDetector

__all__ = ["RiverAnomalyDetector", "SklearnAnomalyDetector"]


def __getattr__(name: str) -> Any:
    if name == "RiverAnomalyDetector":
        from .river_anomaly_detector import RiverAnomalyDetector

        return RiverAnomalyDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
