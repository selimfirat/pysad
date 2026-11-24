from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Literal, cast

import numpy as np
from numpy.typing import NDArray

from pysad.core._iterate import _iterate


class BasePostprocessor(ABC):
    """Base class for postprocessing methods."""

    _output_kind: ClassVar[Literal["score", "probability", "decision"]] = "score"
    _has_decision_rule: ClassVar[bool] = False

    @property
    def output_kind(self) -> Literal["score", "probability", "decision"]:
        """Output kind produced by this postprocessor."""
        return self._output_kind

    def _can_predict(self) -> bool:
        return self.output_kind == "decision" or self._has_decision_rule

    def _decide_from_score(self, score: float) -> int:
        if self.output_kind == "decision":
            return 1 if score > 0.0 else 0
        raise TypeError(
            f"{type(self).__name__} produces continuous outputs (output_kind={self.output_kind!r}) "
            "without an intrinsic decision rule. Pipe it into ThresholdFilter or QuantileFilter "
            "to call predict_partial."
        )

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

    def transform(self, scores: NDArray[Any]) -> NDArray[Any]:
        """Shortcut method that iteratively applies transform_partial to all instances in order.

        Args:
            scores (np.float64 array of shape (num_instances,)): Input scores.

        Returns:
            np.float64 array of shape (num_instances,): Processed scores.
        """
        return self._process_all(scores, self.transform_partial)

    def fit(self, scores: NDArray[Any]) -> BasePostprocessor:
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

    def fit_transform(self, scores: NDArray[Any]) -> NDArray[Any]:
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
    def _process_all(
        scores: NDArray[Any], process_partial: Callable[[float], float]
    ) -> NDArray[Any]:
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
        kind: Literal["score", "probability", "decision"] = "score"
        for step in steps:
            if kind == "decision" and step.output_kind == "probability":
                raise ValueError(
                    f"Cannot apply probability calibrator {type(step).__name__} "
                    "after a binary decision output."
                )
            kind = step.output_kind
        self.steps = steps

    @property
    def output_kind(self) -> Literal["score", "probability", "decision"]:
        return self.steps[-1].output_kind if self.steps else "score"

    def _can_predict(self) -> bool:
        return self.steps[-1]._can_predict() if self.steps else False

    def _decide_from_score(self, score: float) -> int:
        if self.steps:
            return self.steps[-1]._decide_from_score(score)
        return super()._decide_from_score(score)

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

    def fit(self, scores: NDArray[Any]) -> _PostprocessorChain:
        for step in self.steps[:-1]:
            scores = step.fit_transform(scores)
        if self.steps:
            self.steps[-1].fit(scores)
        return self

    def transform(self, scores: NDArray[Any]) -> NDArray[Any]:
        for step in self.steps:
            scores = step.transform(scores)
        return scores

    def fit_transform(self, scores: NDArray[Any]) -> NDArray[Any]:
        for step in self.steps:
            scores = step.fit_transform(scores)
        return scores
