from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from pysad.core.base_model import BaseModel
from pysad.core.base_postprocessor import BasePostprocessor, _PostprocessorChain
from pysad.core.base_transformer import BaseTransformer, _TransformerChain


class StreamPipeline(BaseModel):
    """Chains preprocessors, a detector and postprocessors into one model.

    Each instance goes through the preprocessors in order, then the detector, and then the postprocessors in order. Calibrators, such as :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator`, are postprocessors, so a calibrator goes last in ``postprocessors``. The pipeline is a model, so it has the same methods as a detector and can go where a detector goes, for example into an evaluation loop or next to other detectors in an ensemble.

    ``fit_score_partial`` is the loop of :doc:`/guides/pipelines`: each preprocessor calls ``fit_transform_partial``, the detector calls ``fit_score_partial``, and each postprocessor calls ``fit_transform_partial``. ``fit_partial`` does the same and discards the score, so that the postprocessors learn the scores of the detector. ``score_partial`` calls ``transform_partial`` of each stage and ``score_partial`` of the detector, and fits no stage. A postprocessor such as :class:`~pysad.transform.postprocessing.AveragePostprocessor` returns a statistic of the fitted scores only, so with it ``score_partial`` does not depend on the instance.

    The pipeline fits the stages that you pass, and does not copy them.

    Args:
        model (BaseModel): The detector that scores the preprocessed instances.
        preprocessors (sequence of BaseTransformer): The transformers to apply to each instance, in order (Default=()).
        postprocessors (sequence of BasePostprocessor): The postprocessors and calibrators to apply to each score, in order (Default=()).
    """

    _is_detector: ClassVar[bool] = False

    def __init__(
        self,
        model: BaseModel,
        preprocessors: Sequence[BaseTransformer] = (),
        postprocessors: Sequence[BasePostprocessor] = (),
    ) -> None:
        self.model = model
        self.preprocessors: tuple[BaseTransformer, ...] = tuple(preprocessors)
        self.postprocessors: tuple[BasePostprocessor, ...] = tuple(postprocessors)

    def __or__(self, other: BasePostprocessor) -> StreamPipeline:
        """Appends a postprocessor or postprocessor chain to this pipeline."""
        if isinstance(other, BasePostprocessor):
            p_steps = other.steps if isinstance(other, _PostprocessorChain) else (other,)
            return StreamPipeline(
                self.model,
                preprocessors=self.preprocessors,
                postprocessors=(*self.postprocessors, *p_steps),
            )
        return NotImplemented

    def __ror__(self, other: BaseTransformer) -> StreamPipeline:
        """Prepends a preprocessor or preprocessor chain to this pipeline."""
        if isinstance(other, BaseTransformer):
            t_steps = other.steps if isinstance(other, _TransformerChain) else (other,)
            return StreamPipeline(
                self.model,
                preprocessors=(*t_steps, *self.preprocessors),
                postprocessors=self.postprocessors,
            )
        return NotImplemented

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> StreamPipeline:
        """Updates every stage with the next instance, as ``fit_score_partial`` does.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int or None): The label of the instance, passed to the detector (Default=None).

        Returns:
            object: The pipeline itself.
        """
        self.fit_score_partial(X, y)
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the next instance with every stage, and fits no stage.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The output of the last stage.
        """
        for preprocessor in self.preprocessors:
            X = preprocessor.transform_partial(X)
        score = self.model.score_partial(X)
        for postprocessor in self.postprocessors:
            score = postprocessor.transform_partial(score)

        return score

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Fits every stage with the next instance and returns the output of the last stage.

        The detector scores the instance before it fits it, as in ``BaseModel.fit_score_partial``.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int or None): The label of the instance, passed to the detector (Default=None).

        Returns:
            float: The output of the last stage.
        """
        for preprocessor in self.preprocessors:
            X = preprocessor.fit_transform_partial(X)
        score = self.model.fit_score_partial(X, y)
        for postprocessor in self.postprocessors:
            score = postprocessor.fit_transform_partial(score)

        return score

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> StreamPipeline:
        """Fits every stage to all instances in order.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances to fit, in order.
            y (np.int32 array of shape (num_instances,)): The labels of the instances, passed to the detector (Default=None).

        Returns:
            object: The pipeline itself.
        """
        self.fit_score(X, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Scores all instances in order with every stage, fitting no stage.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances to score, in order.

        Returns:
            np.float64 array of shape (num_instances,): The outputs of the last stage in order.
        """
        for preprocessor in self.preprocessors:
            X = preprocessor.transform(X)
        scores = self.model.score(X)
        for postprocessor in self.postprocessors:
            scores = postprocessor.transform(scores)

        return np.asarray(scores, dtype=np.float64)

    def fit_score(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fits every stage to all instances in order and returns the outputs of the last stage.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): The instances to fit and score, in order.
            y (np.int32 array of shape (num_instances,)): The labels of the instances, passed to the detector (Default=None).

        Returns:
            np.float64 array of shape (num_instances,): The outputs of the last stage in order.
        """
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        for preprocessor in self.preprocessors:
            X = preprocessor.fit_transform(X)
        scores = self.model.fit_score(X, y)
        for postprocessor in self.postprocessors:
            scores = postprocessor.fit_transform(scores)

        return np.asarray(scores, dtype=np.float64)


BaseModel._pipeline_cls = StreamPipeline
