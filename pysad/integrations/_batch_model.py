from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from sklearn.base import OutlierMixin

from pysad.core.base_model import BaseModel


class _BatchModel(BaseModel):
    """Base class for wrappers that run batch anomaly detectors on a stream."""

    _is_detector: ClassVar[bool] = False

    def __init__(self, model_cls: type[Any], **kwargs: Any) -> None:
        self.model_cls = model_cls
        self.kwargs = kwargs
        self.model: Any = None
        self._negate_decision_function = False

    def reset_model(self) -> None:
        """Removes the old model from the memory and instantiates a new one."""
        self.model = self.model_cls(**self.kwargs)
        self._negate_decision_function = isinstance(self.model, OutlierMixin) and not getattr(
            self.model, "_higher_is_anomalous", False
        )

    def _score_one(self, X: np.ndarray) -> float:
        if hasattr(self.model, "score_one"):
            return float(self.model.score_one(X))
        score = float(self.model.decision_function([X])[0])
        return -score if self._negate_decision_function else score

    def _score_batch(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.model, "decision_function") and hasattr(self.model, "score_one"):
            return np.fromiter(
                (float(self.model.score_one(xi)) for xi in X),
                dtype=np.float64,
                count=len(X),
            )
        scores = np.asarray(self.model.decision_function(X), dtype=np.float64).reshape(-1)
        return -scores if self._negate_decision_function else scores
