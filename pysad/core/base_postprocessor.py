from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

import numpy as np
from pysad.utils import _iterate


class BasePostprocessor(ABC):
    """Base class for postprocessing methods.
    """

    @abstractmethod
    def fit_partial(self, score: float) -> "BasePostprocessor":
        """Fits particular (next) timestep's score to train the postprocessor.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        pass

    @abstractmethod
    def transform_partial(self, score: float) -> float:
        """Transforms given score.

        Args:
            score (float): Input score.

        Returns:
            float: Processed score.
        """
        pass

    def fit_transform_partial(self, score: float) -> float:
        """Shortcut method that iteratively applies fit_partial and transform_partial, respectively.

        Args:
            score (float): Input score.

        Returns:
            float: Processed score.
        """
        return self.fit_partial(score).transform_partial(score)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Shortcut method that iteratively applies transform_partial to all instances in order.

        Args:
            np.float64 array of shape (num_instances,): Input scores.

        Returns:
            np.float64 array of shape (num_instances,): Processed scores.
        """
        return self._process_all(scores, self.transform_partial)

    def fit(self, scores: np.ndarray) -> "BasePostprocessor":
        """Shortcut method that iteratively applies fit_partial to all instances in order.

        Args:
            np.float64 array of shape (num_instances,): Input scores.

        Returns:
            object: self.
        """
        for score, _ in _iterate(scores):
            # _iterate yields NumPy scalars for 1-D score arrays, which are floats.
            self.fit_partial(cast(float, score))

        return self

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        """Shortcut method that iteratively applies fit_transform_partial to all instances in order.

        Args:
            np.float64 array of shape (num_instances,): Input scores.

        Returns:
            np.float64 array of shape (num_instances,): Processed scores.
        """
        return self._process_all(scores, self.fit_transform_partial)

    @staticmethod
    def _process_all(scores: np.ndarray, process_partial: Callable[[float], float]) -> np.ndarray:
        processed_scores = np.empty(scores.shape[0], dtype=np.float64)
        for i, (score, _) in enumerate(_iterate(scores)):
            result = process_partial(cast(float, score))
            processed_scores[i] = np.asarray(result).item() if np.asarray(result).ndim > 0 else result

        return processed_scores
