from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_array

from pysad.core.base_model import BaseModel


class SklearnAnomalyDetector(OutlierMixin, BaseEstimator):
    """Expose a fitted PySAD model through the scikit-learn outlier interface.

    Fit the PySAD model before wrapping it. This adapter keeps the same model.
    Its ``fit`` method returns the adapter without learning from ``X`` or ``y``.
    It supports pipelines with fitted preprocessing stages, not model selection.
    Scikit-learn ``clone`` copies the fitted model, rather than resetting it.

    ``score_samples`` returns negative PySAD scores. Lower values mean more anomalous instances.
    ``decision_function`` returns ``threshold - model.score(X)``.
    ``predict`` returns -1 above the threshold and 1 at or below it.
    All scoring methods preserve model state. They keep the model's score range and warm-up behavior.
    If the caller updates the model, later scores reflect those updates.

    Args:
        model (BaseModel): A fitted PySAD model.
        threshold (float or None): Anomaly threshold in PySAD score units. None permits only ``score_samples`` (Default=None).
    """

    def __init__(self, model: BaseModel, threshold: float | None = None) -> None:
        self.model = model
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> SklearnAnomalyDetector:
        """Return this adapter without changing the fitted model.

        Args:
            X (array-like): Ignored input.
            y (array-like or None): Ignored labels (Default=None).

        Returns:
            SklearnAnomalyDetector: This adapter.
        """
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Score a batch without learning from it.

        Args:
            X (array-like of shape (num_instances, num_features)): Numeric instances in the model's feature order.

        Returns:
            np.ndarray: Float64 scores of shape (num_instances,). Lower scores mean more anomalous instances.
        """
        return -self.model.score(check_array(X, dtype=np.float64))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return signed distances from the anomaly threshold.

        Args:
            X (array-like of shape (num_instances, num_features)): Instances to score.

        Returns:
            np.ndarray: Negative values for outliers. Zero and positive values for inliers.

        Raises:
            ValueError: If threshold is None.
        """
        if self.threshold is None:
            raise ValueError("Set threshold before calling decision_function or predict.")
        return self.score_samples(X) + self.threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Label instances with the anomaly threshold.

        Args:
            X (array-like of shape (num_instances, num_features)): Instances to label.

        Returns:
            np.ndarray: Integer labels of shape (num_instances,). Outliers use -1 and inliers use 1.

        Raises:
            ValueError: If threshold is None.
        """
        return np.where(self.decision_function(X) < 0.0, -1, 1)
