from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

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
        kind = model.output_kind
        for postprocessor in postprocessors:
            if kind == "decision" and postprocessor.output_kind == "probability":
                raise ValueError(
                    f"Cannot apply probability calibrator {type(postprocessor).__name__} "
                    "after a binary decision output."
                )
            kind = postprocessor.output_kind
        self.model = model
        self.preprocessors: tuple[BaseTransformer, ...] = tuple(preprocessors)
        self.postprocessors: tuple[BasePostprocessor, ...] = tuple(postprocessors)

    @property
    def output_kind(self) -> Literal["score", "probability", "decision"]:
        """Output kind produced by the final stage of the pipeline."""
        if self.postprocessors:
            return self.postprocessors[-1].output_kind
        return self.model.output_kind

    def _can_predict(self) -> bool:
        if self.postprocessors:
            return self.postprocessors[-1]._can_predict()
        return self.model._can_predict()

    def _decide_from_score(self, score: float) -> int:
        if self.postprocessors:
            return self.postprocessors[-1]._decide_from_score(score)
        return self.model._decide_from_score(score)

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

    def fit_partial(self, X: Any, y: int | None = None) -> StreamPipeline:
        """Updates every stage with the next instance, as ``fit_score_partial`` does.

        Args:
            X (np.float64 array of shape (num_features,) or Any): The instance to fit.
            y (int or None): The label of the instance, passed to the detector (Default=None).

        Returns:
            object: The pipeline itself.
        """
        self.fit_score_partial(X, y)
        return self

    def score_partial(self, X: Any) -> float:
        """Scores the next instance with every stage, and fits no stage.

        Args:
            X (np.float64 array of shape (num_features,) or Any): The instance to score.

        Returns:
            float: The output of the last stage. Higher scores represent more anomalous
                instances, consistent with the output kind of the last model or postprocessor
                in the pipeline.
        """
        for preprocessor in self.preprocessors:
            X = preprocessor.transform_partial(X)
        score = self.model.score_partial(X)
        for postprocessor in self.postprocessors:
            score = postprocessor.transform_partial(score)

        return score

    def fit_score_partial(self, X: Any, y: int | None = None) -> float:
        """Fits every stage with the next instance and returns the output of the last stage.

        The detector scores the instance before it fits it, as in ``BaseModel.fit_score_partial``.

        Args:
            X (np.float64 array of shape (num_features,) or Any): The instance to fit and score.
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

    def fit(self, X: Any, y: NDArray[Any] | None = None) -> StreamPipeline:
        """Fits every stage to all instances in order.

        Args:
            X (np.float64 array of shape (num_instances, num_features) or Sequence[Any]): The instances to fit, in order.
            y (np.int32 array of shape (num_instances,)): The labels of the instances, passed to the detector (Default=None).

        Returns:
            object: The pipeline itself.
        """
        self.fit_score(X, y)
        return self

    def score(self, X: Any) -> NDArray[Any]:
        """Scores all instances in order with every stage, fitting no stage.

        Args:
            X (np.float64 array of shape (num_instances, num_features) or Sequence[Any]): The instances to score, in order.

        Returns:
            np.float64 array of shape (num_instances,): The outputs of the last stage in order.
        """
        for preprocessor in self.preprocessors:
            X = preprocessor.transform(X)
        scores = self.model.score(X)
        for postprocessor in self.postprocessors:
            scores = postprocessor.transform(scores)

        return np.asarray(scores, dtype=np.float64)

    def fit_score(self, X: Any, y: NDArray[Any] | None = None) -> NDArray[Any]:
        """Fits every stage to all instances in order and returns the outputs of the last stage.

        Args:
            X (np.float64 array of shape (num_instances, num_features) or Sequence[Any]): The instances to fit and score, in order.
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

    def predict_partial(self, X: Any) -> int:
        """Predicts a binary anomaly label for one instance without fitting any stage."""
        if not self._can_predict():
            self._decide_from_score(0.0)
        for preprocessor in self.preprocessors:
            X = preprocessor.transform_partial(X)
        if not self.postprocessors:
            return self.model.predict_partial(X)
        score = self.model.score_partial(X)
        for postprocessor in self.postprocessors:
            score = postprocessor.transform_partial(score)
        return self.postprocessors[-1]._decide_from_score(score)

    def fit_predict_partial(self, X: Any, y: int | None = None) -> int:
        """Fits every stage on one instance and returns a binary anomaly decision."""
        if not self._can_predict():
            self._decide_from_score(0.0)
        for preprocessor in self.preprocessors:
            X = preprocessor.fit_transform_partial(X)
        if not self.postprocessors:
            return self.model.fit_predict_partial(X, y)
        score = self.model.fit_score_partial(X, y)
        for postprocessor in self.postprocessors:
            score = postprocessor.fit_transform_partial(score)
        return self.postprocessors[-1]._decide_from_score(score)

    def predict(self, X: Any) -> NDArray[np.int64]:
        """Predicts binary anomaly labels for all instances in order without fitting."""
        if not self._can_predict():
            self._decide_from_score(0.0)
        for preprocessor in self.preprocessors:
            X = preprocessor.transform(X)
        if not self.postprocessors:
            return self.model.predict(X)
        scores = self.model.score(X)
        for postprocessor in self.postprocessors:
            scores = postprocessor.transform(scores)
        last = self.postprocessors[-1]
        if last.output_kind == "decision":
            return np.asarray(scores > 0.0, dtype=np.int64)
        decide = last._decide_from_score
        return np.array([decide(float(s)) for s in scores], dtype=np.int64)

    def fit_predict(self, X: Any, y: NDArray[Any] | None = None) -> NDArray[np.int64]:
        """Fits every stage on all instances in order and returns binary anomaly labels."""
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if not self._can_predict():
            self._decide_from_score(0.0)
        for preprocessor in self.preprocessors:
            X = preprocessor.fit_transform(X)
        if not self.postprocessors:
            return self.model.fit_predict(X, y)
        scores = self.model.fit_score(X, y)
        for postprocessor in self.postprocessors:
            scores = postprocessor.fit_transform(scores)
        last = self.postprocessors[-1]
        if last.output_kind == "decision":
            return np.asarray(scores > 0.0, dtype=np.int64)
        decide = last._decide_from_score
        return np.array([decide(float(s)) for s in scores], dtype=np.int64)


BaseModel._pipeline_cls = StreamPipeline
