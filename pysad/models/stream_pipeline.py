from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pysad.core.base_model import BaseModel
from pysad.core.base_postprocessor import BasePostprocessor
from pysad.core.base_transformer import BaseTransformer


class StreamPipeline(BaseModel):
    """Chains preprocessors, a detector and postprocessors into one model.

    Each instance goes through the preprocessors in order, then the detector, and then the postprocessors in order. Calibrators, such as :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator`, are postprocessors, so a calibrator goes last in ``postprocessors``. The pipeline is a model, so it has the same methods as a detector and can go where a detector goes, for example into an evaluation loop or next to other detectors in an ensemble.

    ``fit_score_partial`` is the loop of :doc:`/guides/pipelines`: each preprocessor calls ``fit_transform_partial``, the detector calls ``fit_score_partial``, and each postprocessor calls ``fit_transform_partial``. ``fit_partial`` does the same and discards the score, so that the postprocessors learn the scores of the detector. ``score_partial`` calls ``transform_partial`` of each stage and ``score_partial`` of the detector, and fits no stage. A postprocessor such as :class:`~pysad.transform.postprocessing.RunningAveragePostprocessor` returns a statistic of the fitted scores only, so with it ``score_partial`` does not depend on the instance.

    The pipeline fits the stages that you pass, and does not copy them.

    Args:
        model (BaseModel): The detector that scores the preprocessed instances.
        preprocessors (sequence of BaseTransformer): The transformers to apply to each instance, in order (Default=()).
        postprocessors (sequence of BasePostprocessor): The postprocessors and calibrators to apply to each score, in order (Default=()).
    """

    def __init__(
        self,
        model: BaseModel,
        preprocessors: Sequence[BaseTransformer] = (),
        postprocessors: Sequence[BasePostprocessor] = (),
    ) -> None:
        self.model = model
        self.preprocessors = preprocessors
        self.postprocessors = postprocessors

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
