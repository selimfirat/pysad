from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pysad.core.base_model import BaseModel


class PySADToScikitMultiflowAdapter:
    """Expose a PySAD model through scikit-multiflow's streaming estimator interface.

    This is a zero-dependency adapter: it never imports scikit-multiflow. It duck-types
    the scikit-multiflow stream estimator interface (``partial_fit``, ``predict``,
    ``predict_proba`` and ``score``), so any PySAD model can be used in scikit-multiflow
    streams and evaluators without installing scikit-multiflow.

    ``partial_fit(X, y)`` learns each row of ``X`` with the wrapped model's ``fit_partial``.
    ``y`` is optional and ignored by unsupervised PySAD models. Extra scikit-multiflow
    arguments such as ``classes``, ``sample_weight`` or ``weight`` are accepted for
    interface compatibility and ignored.

    ``predict(X)`` returns binary anomaly labels per row (0 normal, 1 anomalous) from the
    wrapped model's ``predict_partial``. It requires the wrapped model to expose binary
    decisions (``output_kind == "decision"`` or a paper-defined decision rule); otherwise it
    raises ``TypeError``.

    ``predict_proba(X)`` returns one row ``[P(normal), P(anomaly)]`` per instance. It is only
    available for probability-calibrated models (``output_kind == "probability"``), where the
    model score is the anomaly probability; otherwise it raises ``ValueError``.

    ``score(X, y)`` returns the fraction of rows whose predicted label equals ``y``.

    Args:
        model (BaseModel): A PySAD model to expose through the scikit-multiflow interface.
    """

    def __init__(self, model: BaseModel) -> None:
        self.model = model

    def partial_fit(
        self,
        X: NDArray[Any],
        y: NDArray[Any] | None = None,
        classes: NDArray[Any] | None = None,
        sample_weight: NDArray[Any] | None = None,
        **kwargs: Any,
    ) -> PySADToScikitMultiflowAdapter:
        """Learns each row of ``X`` and returns the adapter.

        Args:
            X (array-like of shape (num_instances, num_features)): Instances to learn from.
            y (array-like of shape (num_instances,) or None): Instance labels. PySAD models
                are unsupervised and ignore it (Default=None).
            classes (array-like or None): Class labels. Accepted for scikit-multiflow
                interface compatibility and ignored (Default=None).
            sample_weight (array-like or None): Instance weights. Accepted for
                scikit-multiflow interface compatibility and ignored (Default=None).
            **kwargs: Additional scikit-multiflow keyword arguments, such as ``weight``,
                accepted for interface compatibility and ignored.

        Returns:
            PySADToScikitMultiflowAdapter: This adapter.

        Raises:
            ValueError: If ``X`` and ``y`` have different numbers of rows, or if ``X`` is not
                a one- or two-dimensional array.
        """
        _ = (classes, sample_weight, kwargs)
        X_arr = self._as_2d_array(X)
        if y is None:
            y_arr: NDArray[Any] = np.zeros(X_arr.shape[0], dtype=np.float64)
        else:
            y_arr = np.asarray(y, dtype=np.float64).ravel()
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows, "
                f"got {X_arr.shape[0]} and {y_arr.shape[0]}."
            )
        fit_fn = self.model.fit_partial
        for i in range(X_arr.shape[0]):
            fit_fn(X_arr[i], y_arr[i])
        return self

    def predict(self, X: NDArray[Any]) -> NDArray[np.int64]:
        """Predicts binary anomaly labels per row without learning.

        Args:
            X (array-like of shape (num_instances, num_features)): Instances to label.

        Returns:
            np.int64 array of shape (num_instances,): Binary labels, 0 for normal and
                1 for anomalous.

        Raises:
            TypeError: If the wrapped model has no binary output or decision rule.
            ValueError: If ``X`` is not a one- or two-dimensional array.
        """
        if not self.model._can_predict():
            raise TypeError(
                f"{type(self.model).__name__} produces continuous outputs "
                f"(output_kind={self.model.output_kind!r}) without an intrinsic decision "
                "rule. Pipe it into ThresholdFilter or QuantileFilter to get binary labels."
            )
        X_arr = self._as_2d_array(X)
        preds = np.empty(X_arr.shape[0], dtype=np.int64)
        for i in range(X_arr.shape[0]):
            preds[i] = self.model.predict_partial(X_arr[i])
        return preds

    def predict_proba(self, X: NDArray[Any]) -> NDArray[np.float64]:
        """Returns anomaly probabilities per row as ``[P(normal), P(anomaly)]``.

        Only probability-calibrated models (``output_kind == "probability"``) are supported;
        the model score is taken as the anomaly probability.

        Args:
            X (array-like of shape (num_instances, num_features)): Instances to score.

        Returns:
            np.float64 array of shape (num_instances, 2): Column 0 holds P(normal) and
                column 1 holds P(anomaly).

        Raises:
            ValueError: If the wrapped model does not output probabilities, or if ``X`` is
                not a one- or two-dimensional array.
        """
        if self.model.output_kind != "probability":
            raise ValueError(
                f"{type(self.model).__name__} has output_kind {self.model.output_kind!r}; "
                "predict_proba is only available for probability-calibrated models."
            )
        X_arr = self._as_2d_array(X)
        proba = np.empty((X_arr.shape[0], 2), dtype=np.float64)
        for i in range(X_arr.shape[0]):
            score = float(self.model.score_partial(X_arr[i]))
            proba[i, 0] = 1.0 - score
            proba[i, 1] = score
        return proba

    def score(self, X: NDArray[Any], y: NDArray[Any]) -> float:
        """Returns the fraction of rows whose predicted label equals ``y``.

        Args:
            X (array-like of shape (num_instances, num_features)): Instances to label.
            y (array-like of shape (num_instances,)): True binary labels, 0 or 1.

        Returns:
            float: Accuracy of the predicted labels, between 0.0 and 1.0.

        Raises:
            TypeError: If the wrapped model has no binary output or decision rule.
            ValueError: If ``X`` and ``y`` have different numbers of rows, or if ``X`` is
                not a one- or two-dimensional array.
        """
        y_arr = np.asarray(y).ravel()
        preds = self.predict(X)
        if y_arr.shape[0] != preds.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows, "
                f"got {preds.shape[0]} and {y_arr.shape[0]}."
            )
        return float(np.mean(preds == y_arr))

    @staticmethod
    def _as_2d_array(X: Any) -> NDArray[Any]:
        """Converts ``X`` to a two-dimensional float64 array.

        A one-dimensional array is treated as a single instance.

        Args:
            X (array-like): One or two dimensional numeric input.

        Returns:
            np.ndarray: Float64 array of shape (num_instances, num_features).

        Raises:
            ValueError: If ``X`` has more than two dimensions.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"X must be a one- or two-dimensional array, got shape {arr.shape}.")
        return arr
