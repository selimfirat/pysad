from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np
from river import base

from pysad.core.base_model import BaseModel


class RiverAnomalyDetector(base.AnomalyDetector):
    """Exposes a PySAD model through River's anomaly detector interface.

    Requires Python 3.11 or newer and River 0.26.1 or newer.

    ``learn_one`` updates the wrapped model with ``fit_partial``; ``score_one``
    calls ``score_partial`` without learning. Higher scores mean more anomalous
    instances, with the wrapped model's score range and warm-up behavior unchanged.
    In a River pipeline, call ``score_one`` before ``learn_one`` for each instance.

    Dictionary values are converted to a float64 array in a fixed feature order.
    By default, the first call to either method records the dictionary's key order
    in ``feature_names_``. Later dictionaries may have a different insertion order,
    but must contain exactly the same keys. Missing or new features raise
    ``ValueError`` before the wrapped model is called. Feature names need not be
    sortable. Scoring may establish the feature order, but never fits the model.

    ``model_factory`` must return a new, untrained PySAD model on each call. Use a
    model class for default parameters, or ``functools.partial`` for custom ones.
    The created model is available as ``model_``. River's default ``clone()``
    creates an untrained model and resets inferred feature names;
    ``clone(include_attributes=True)`` copies the learned model and feature order.

    Args:
        model_factory (callable): A zero-argument callable that creates a fresh PySAD model.
        feature_names (sequence of hashable or None): Feature names in the model's column order. Names must be unique. If None, use the first dictionary's key order (Default=None).
    """

    def __init__(
        self,
        model_factory: Callable[[], BaseModel],
        feature_names: Sequence[Hashable] | None = None,
    ) -> None:
        self.model_factory = model_factory
        self.model_ = model_factory()
        self.feature_names = None if feature_names is None else tuple(feature_names)
        self.feature_names_ = self.feature_names
        if self.feature_names is not None and len(self.feature_names) != len(
            set(self.feature_names)
        ):
            raise ValueError("feature_names must contain unique names.")
        self._feature_keys = None if feature_names is None else frozenset(feature_names)

    def learn_one(self, x: dict[Any, Any]) -> None:
        """Learns one feature dictionary without scoring it.

        Args:
            x (dict): Numeric feature values with the fixed feature names.
        """
        self.model_.fit_partial(self._to_array(x))

    def score_one(self, x: dict[Any, Any]) -> float:
        """Scores one feature dictionary without learning from it.

        Args:
            x (dict): Numeric feature values with the fixed feature names.

        Returns:
            float: The wrapped model's anomaly score.
        """
        return self.model_.score_partial(self._to_array(x))

    def _to_array(self, x: dict[Any, Any]) -> np.ndarray:
        if self.feature_names_ is None:
            self.feature_names_ = tuple(x)
            self._feature_keys = frozenset(x)
        if x.keys() != self._feature_keys:
            raise ValueError("Expected exactly the same feature names for every instance.")
        return np.fromiter(
            (x[name] for name in self.feature_names_),
            dtype=np.float64,
            count=len(self.feature_names_),
        )
