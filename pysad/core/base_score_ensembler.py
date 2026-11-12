from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BaseScoreEnsembler(ABC):
    """Abstract base class for streaming multi-detector score ensemblers.

    Single-timestep methods take a 1-D vector of ``num_detectors`` scores and return a ``float``.
    Batch methods take a 2-D matrix of shape ``(num_instances, num_detectors)`` and return a 1-D
    ``np.float64`` array of shape ``(num_instances,)``. Higher combined scores mean more anomalous
    instances.
    """

    _stateless: ClassVar[bool] = False

    @staticmethod
    def _as_2d_scores(scores: ArrayLike) -> NDArray[np.float64]:
        arr = np.asarray(scores, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"Expected a 2-D score matrix of shape (num_instances, num_detectors), got {arr.shape}."
            )
        return arr

    @abstractmethod
    def fit_partial(self, scores: ArrayLike) -> BaseScoreEnsembler:
        """Updates the ensembler with one timestep's vector of detector scores.

        Args:
            scores (array-like of shape (num_detectors,)): Anomaly scores from each detector.

        Returns:
            BaseScoreEnsembler: ``self``.
        """
        pass

    @abstractmethod
    def transform_partial(self, scores: ArrayLike) -> float:
        """Combines one timestep's vector of detector scores without mutating state.

        Args:
            scores (array-like of shape (num_detectors,)): Anomaly scores from each detector.

        Returns:
            float: Combined anomaly score.
        """
        pass

    def fit_transform_partial(self, scores: ArrayLike) -> float:
        """Updates the ensembler with ``scores`` and returns the combined score.

        Args:
            scores (array-like of shape (num_detectors,)): Anomaly scores from each detector.

        Returns:
            float: Combined anomaly score.
        """
        self.fit_partial(scores)
        res = self.transform_partial(scores)
        return res if type(res) is float else float(res)

    def fit(self, scores: np.ndarray) -> BaseScoreEnsembler:
        """Fits the ensembler to all rows of ``scores`` in order.

        Args:
            scores (np.float64 array of shape (num_instances, num_detectors)): Detector scores per instance.

        Returns:
            BaseScoreEnsembler: ``self``.
        """
        arr = self._as_2d_scores(scores)
        for i in range(len(arr)):
            self.fit_partial(arr[i])
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Combines each row of ``scores`` in order using ``transform_partial``.

        Args:
            scores (np.float64 array of shape (num_instances, num_detectors)): Detector scores per instance.

        Returns:
            np.float64 array of shape (num_instances,): Combined anomaly scores.
        """
        arr = self._as_2d_scores(scores)
        out = np.empty(len(arr), dtype=np.float64)
        for i in range(len(arr)):
            res = self.transform_partial(arr[i])
            out[i] = res if type(res) is float else float(res)
        return out

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        """Applies ``fit_transform_partial`` to each row of ``scores`` in order.

        Args:
            scores (np.float64 array of shape (num_instances, num_detectors)): Detector scores per instance.

        Returns:
            np.float64 array of shape (num_instances,): Combined anomaly scores.
        """
        arr = self._as_2d_scores(scores)
        out = np.empty(len(arr), dtype=np.float64)
        for i in range(len(arr)):
            out[i] = self.fit_transform_partial(arr[i])
        return out
