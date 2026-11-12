from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

import numpy as np

from pysad.utils import _iterate


class BasePostprocessor(ABC):
    """Base class for postprocessing methods."""

    @abstractmethod
    def fit_partial(self, score: float) -> BasePostprocessor:
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
        """Updates the postprocessor with the next score and transforms the score.

        Args:
            score (float): Input score.

        Returns:
            float: Processed score.
        """
        return self.fit_partial(score).transform_partial(score)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Shortcut method that iteratively applies transform_partial to all instances in order.

        Args:
            scores (np.float64 array of shape (num_instances,)): Input scores.

        Returns:
            np.float64 array of shape (num_instances,): Processed scores.
        """
        return self._process_all(scores, self.transform_partial)

    def fit(self, scores: np.ndarray) -> BasePostprocessor:
        """Shortcut method that iteratively applies fit_partial to all instances in order.

        Args:
            scores (np.float64 array of shape (num_instances,)): Input scores.

        Returns:
            object: self.
        """
        if isinstance(scores, np.ndarray) and scores.ndim == 1:
            fit_fn = self.fit_partial
            for i in range(len(scores)):
                fit_fn(float(scores[i]))
            return self

        for score, _ in _iterate(scores):
            # _iterate yields NumPy scalars for 1-D score arrays, which are floats.
            self.fit_partial(cast(float, score))

        return self

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        """Shortcut method that iteratively applies fit_transform_partial to all instances in order.

        Args:
            scores (np.float64 array of shape (num_instances,)): Input scores.

        Returns:
            np.float64 array of shape (num_instances,): Processed scores.
        """
        return self._process_all(scores, self.fit_transform_partial)

    def __or__(self, other: BasePostprocessor) -> _PostprocessorChain:
        """Chains this postprocessor with another postprocessor."""
        if isinstance(other, BasePostprocessor):
            left = self.steps if isinstance(self, _PostprocessorChain) else (self,)
            right = other.steps if isinstance(other, _PostprocessorChain) else (other,)
            return _PostprocessorChain((*left, *right))
        return NotImplemented

    @staticmethod
    def _process_all(scores: np.ndarray, process_partial: Callable[[float], float]) -> np.ndarray:
        processed_scores = np.empty(scores.shape[0], dtype=np.float64)
        if isinstance(scores, np.ndarray) and scores.ndim == 1:
            for i in range(len(scores)):
                res = process_partial(float(scores[i]))
                processed_scores[i] = res if type(res) is float else float(np.asarray(res).item())
            return processed_scores

        for i, (score, _) in enumerate(_iterate(scores)):
            result = process_partial(cast(float, score))
            processed_scores[i] = (
                result if type(result) is float else float(np.asarray(result).item())
            )

        return processed_scores


class _PostprocessorChain(BasePostprocessor):
    """Lightweight sequential chain of postprocessors created by ``|``."""

    def __init__(self, steps: tuple[BasePostprocessor, ...]) -> None:
        self.steps = steps

    def fit_partial(self, score: float) -> _PostprocessorChain:
        for step in self.steps[:-1]:
            score = step.fit_transform_partial(score)
        if self.steps:
            self.steps[-1].fit_partial(score)
        return self

    def transform_partial(self, score: float) -> float:
        for step in self.steps:
            score = step.transform_partial(score)
        return score

    def fit_transform_partial(self, score: float) -> float:
        for step in self.steps:
            score = step.fit_transform_partial(score)
        return score

    def fit(self, scores: np.ndarray) -> _PostprocessorChain:
        for step in self.steps[:-1]:
            scores = step.fit_transform(scores)
        if self.steps:
            self.steps[-1].fit(scores)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        for step in self.steps:
            scores = step.transform(scores)
        return scores

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        for step in self.steps:
            scores = step.fit_transform(scores)
        return scores
