from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pysad.core.base_model import BaseModel


class PySADToCapyMoaAdapter:
    """Expose a PySAD detector through CapyMoa's streaming detector interface.

    This is a zero-dependency adapter: it never imports CapyMoa. It duck-types CapyMoa's
    streaming anomaly detector interface (``partial_fit``, ``score``, ``predict``,
    ``predict_proba`` and ``reset``), so any PySAD model can be used in CapyMoa streams
    and evaluators without installing CapyMoa.

    ``partial_fit(instance, y=None)`` learns one instance from ``instance.x`` with the
    wrapped model's ``fit_partial`` and returns the adapter. An optional ``y`` is accepted
    for interface compatibility and ignored by unsupervised PySAD models.

    ``score(instance)`` returns the anomaly score from ``score_partial`` without learning.
    Higher scores mean more anomalous instances.

    ``predict(instance)`` returns a binary label, 0 for normal and 1 for anomalous, from
    ``predict_partial``. It requires the wrapped model to expose binary decisions
    (``output_kind == "decision"`` or a paper-defined decision rule); otherwise it raises
    ``TypeError``.

    ``predict_proba(instance)`` returns ``[P(normal), P(anomaly)]``. It is only available
    for probability-calibrated models (``output_kind == "probability"``); otherwise it raises
    ``ValueError``.

    ``reset()`` is a no-op: PySAD models do not expose a reset operation. Build a fresh
    adapter around a fresh model to start over.

    Instance objects are duck-typed: anything with an ``x`` attribute holding the feature
    vector works. A raw one-dimensional NumPy array or scalar is also accepted, which is
    useful when CapyMoa is not installed.

    Args:
        model (BaseModel): A PySAD detector to expose through CapyMoa's interface.
    """

    def __init__(self, model: BaseModel) -> None:
        self.model = model

    def partial_fit(self, instance: Any, y: Any = None) -> PySADToCapyMoaAdapter:
        """Learns one instance and returns the adapter.

        Args:
            instance: A CapyMoa-like instance with an ``x`` feature vector, or a raw
                one-dimensional numeric array.
            y: The instance label. Accepted for interface compatibility and ignored
                (Default=None).

        Returns:
            PySADToCapyMoaAdapter: This adapter.

        Raises:
            ValueError: If the instance does not hold a one-dimensional feature vector.
        """
        _ = y
        x = self._extract_features(instance)
        self.model.fit_partial(x)
        return self

    def score(self, instance: Any) -> float:
        """Returns the anomaly score of one instance without learning.

        Args:
            instance: A CapyMoa-like instance with an ``x`` feature vector, or a raw
                one-dimensional numeric array.

        Returns:
            float: The wrapped model's anomaly score.

        Raises:
            ValueError: If the instance does not hold a one-dimensional feature vector.
        """
        x = self._extract_features(instance)
        return self.model.score_partial(x)

    def predict(self, instance: Any) -> int:
        """Returns a binary label for one instance, 0 for normal and 1 for anomalous.

        Args:
            instance: A CapyMoa-like instance with an ``x`` feature vector, or a raw
                one-dimensional numeric array.

        Returns:
            int: ``0`` for normal and ``1`` for anomalous.

        Raises:
            TypeError: If the wrapped model has no binary output or decision rule.
            ValueError: If the instance does not hold a one-dimensional feature vector.
        """
        if not self.model._can_predict():
            raise TypeError(
                f"{type(self.model).__name__} produces continuous outputs "
                f"(output_kind={self.model.output_kind!r}) without an intrinsic decision "
                "rule. Pipe it into ThresholdFilter or QuantileFilter to get binary labels."
            )
        x = self._extract_features(instance)
        return self.model.predict_partial(x)

    def predict_proba(self, instance: Any) -> NDArray[np.float64]:
        """Returns anomaly probabilities as ``[P(normal), P(anomaly)]``.

        Only probability-calibrated models (``output_kind == "probability"``) are supported;
        the model score is taken as the anomaly probability.

        Args:
            instance: A CapyMoa-like instance with an ``x`` feature vector, or a raw
                one-dimensional numeric array.

        Returns:
            np.float64 array of shape (2,): Column 0 holds P(normal) and column 1 holds
                P(anomaly).

        Raises:
            ValueError: If the wrapped model does not output probabilities, or if the
                instance does not hold a one-dimensional feature vector.
        """
        if self.model.output_kind != "probability":
            raise ValueError(
                f"{type(self.model).__name__} has output_kind {self.model.output_kind!r}; "
                "predict_proba is only available for probability-calibrated models."
            )
        x = self._extract_features(instance)
        score = float(self.model.score_partial(x))
        return np.array([1.0 - score, score], dtype=np.float64)

    def reset(self) -> None:
        """Resets the adapter.

        PySAD models do not expose a reset operation, so this is a no-op. Build a fresh
        adapter around a fresh model to start over.
        """
        return None

    @staticmethod
    def _extract_features(instance: Any) -> NDArray[np.float64]:
        """Extracts the one-dimensional feature vector from a CapyMoa-like instance.

        A CapyMoa ``Instance`` exposes its features in ``.x``. Anything with an ``x``
        attribute is treated as an instance; otherwise the argument itself is used as the
        feature vector.

        Args:
            instance: A CapyMoa-like instance or a raw one-dimensional numeric array.

        Returns:
            np.float64 array of shape (num_features,).

        Raises:
            ValueError: If the features are not a one-dimensional vector.
        """
        x = getattr(instance, "x", None)
        if x is None:
            x = instance
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError(f"Expected a one-dimensional feature vector, got shape {arr.shape}.")
        return arr
