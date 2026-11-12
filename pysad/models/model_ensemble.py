from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from pysad.core.base_model import BaseModel
from pysad.core.base_score_ensembler import BaseScoreEnsembler
from pysad.models.stream_pipeline import StreamPipeline
from pysad.transform.ensemble import AverageScoreEnsembler


def _uses_global_rng(obj: Any) -> bool:
    """Returns whether ``obj`` may draw from NumPy's global ``RandomState`` during streaming."""
    if isinstance(obj, ModelEnsemble):
        return any(_uses_global_rng(m) for m in obj.models) or _uses_global_rng(obj.ensembler)
    if isinstance(obj, StreamPipeline):
        return (
            any(_uses_global_rng(t) for t in obj.preprocessors)
            or _uses_global_rng(obj.model)
            or any(_uses_global_rng(p) for p in obj.postprocessors)
        )
    global_rng = np.random.mtrand._rand
    for val in vars(obj).values():
        if val is global_rng:
            return True
    return getattr(obj, "random_state", 0) is None and (
        "random_state" in inspect.signature(type(obj).__init__).parameters
    )


class ModelEnsemble(BaseModel):
    """Combines multiple streaming anomaly detectors (or pipelines) with a score ensembler.

    Each instance is passed to every model in ``models`` in order, and the resulting vector of
    anomaly scores is combined by ``ensembler`` (defaulting to :class:`~pysad.transform.ensemble.AverageScoreEnsembler`).
    Because raw detector scores have different scales, wrap each detector in a normalizing pipeline
    first—for example ``(LODA() | ZScorePostprocessor(200)) & (xStream() | ZScorePostprocessor(200))``.

    ``ModelEnsemble`` is itself a :class:`~pysad.core.BaseModel`, so it composes with preprocessors
    and calibrators via ``|`` and works directly with :func:`~pysad.evaluation.evaluate_stream`.

    Args:
        models (sequence of BaseModel): At least one detector or pipeline to ensemble, in order.
        ensembler (BaseScoreEnsembler or None): Score combiner applied to the vector of model scores. If None, uses :class:`~pysad.transform.ensemble.AverageScoreEnsembler` (Default=None).

    Raises:
        ValueError: If ``models`` is empty.
    """

    _is_detector: ClassVar[bool] = False

    def __init__(
        self,
        models: Sequence[BaseModel],
        ensembler: BaseScoreEnsembler | None = None,
    ) -> None:
        if len(models) == 0:
            raise ValueError("models must contain at least one model.")
        self.models = tuple(models)
        self._default_ensembler = ensembler is None
        self.ensembler: BaseScoreEnsembler = (
            AverageScoreEnsembler() if ensembler is None else ensembler
        )

    def _can_batch(self) -> bool:
        return sum(_uses_global_rng(c) for c in (*self.models, self.ensembler)) <= 1

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> ModelEnsemble:
        """Updates every constituent model (and the ensembler if stateful) with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int or None): The label of the instance, passed to each model (Default=None).

        Returns:
            ModelEnsemble: The ensemble itself.
        """
        if self.ensembler._stateless:
            for model in self.models:
                model.fit_partial(X, y)
            return self

        scores = np.fromiter(
            (model.fit_score_partial(X, y) for model in self.models),
            dtype=np.float64,
            count=len(self.models),
        )
        self.ensembler.fit_partial(scores)
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the next instance across all constituent models without mutating state.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: Combined anomaly score.
        """
        scores = np.fromiter(
            (model.score_partial(X) for model in self.models),
            dtype=np.float64,
            count=len(self.models),
        )
        return self.ensembler.transform_partial(scores)

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Fits and scores the next instance across all constituent models in prequential order.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int or None): The label of the instance, passed to each model (Default=None).

        Returns:
            float: Combined anomaly score.
        """
        scores = np.fromiter(
            (model.fit_score_partial(X, y) for model in self.models),
            dtype=np.float64,
            count=len(self.models),
        )
        return self.ensembler.fit_transform_partial(scores)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ModelEnsemble:
        """Fits all constituent models (and the ensembler if stateful) to a batch of instances."""
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if not self._can_batch():
            super().fit(X, y)
            return self
        if len(X) == 0:
            return self
        if self.ensembler._stateless:
            for model in self.models:
                model.fit(X, y)
            return self
        self.fit_score(X, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Scores a batch of instances across all constituent models without mutating state."""
        if not self._can_batch():
            return super().score(X)
        if len(X) == 0:
            return np.empty(0, dtype=np.float64)
        matrix = np.column_stack([model.score(X) for model in self.models])
        return np.asarray(self.ensembler.transform(matrix), dtype=np.float64)

    def fit_score(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fits and scores a batch of instances across all constituent models in order."""
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if not self._can_batch():
            return super().fit_score(X, y)
        if len(X) == 0:
            return np.empty(0, dtype=np.float64)
        matrix = np.column_stack([model.fit_score(X, y) for model in self.models])
        return np.asarray(self.ensembler.fit_transform(matrix), dtype=np.float64)


BaseModel._ensemble_cls = ModelEnsemble
