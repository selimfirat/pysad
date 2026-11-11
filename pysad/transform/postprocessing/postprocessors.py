from __future__ import annotations

from typing import cast

import numpy as np

from pysad.core.base_postprocessor import BasePostprocessor
from pysad.core.base_statistic import UnivariateStatistic
from pysad.statistics.average_meter import AverageMeter
from pysad.statistics.max_meter import MaxMeter
from pysad.statistics.quantile_meter import QuantileMeter
from pysad.statistics.running_statistic import RunningStatistic
from pysad.statistics.variance_meter import VarianceMeter


class _MeterPostprocessor(BasePostprocessor):
    """Base class for postprocessors that convert a score to a statistic of the previous scores.

    Args:
        meter: The statistic to update with each score.
    """

    def __init__(self, meter: UnivariateStatistic | RunningStatistic) -> None:
        self.meter = meter
        self.num_seen_ = 0

    def fit_partial(self, score: float) -> _MeterPostprocessor:
        """Fits the postprocessor to the (next) timestep's score. Running postprocessors only keep the scores in their window.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.meter.update(score)
        self.num_seen_ += 1

        return self

    def transform_partial(self, score: float | None = None) -> float:
        """Applies postprocessing to the score, using only the scores in the window for running postprocessors. This method should be used immediately after the fit_partial method with same score.

        Args:
            score (float): The input score.

        Returns:
            float: Transformed score, or 0.0 before the first fit.
        """
        if self.num_seen_ == 0:
            return 0.0
        return self.meter.get()

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Transforms a batch of scores using the current frozen meter statistic."""
        n = len(scores)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        val = 0.0 if self.num_seen_ == 0 else float(self.meter.get())
        return np.full(n, val, dtype=np.float64)


class _ZScorePostprocessor(BasePostprocessor):
    """Base class for postprocessors that normalize the score via Z-score normalization.

    Args:
        variance_meter: The variance statistic of the previous scores.
        average_meter: The average statistic of the previous scores.
    """

    def __init__(
        self,
        variance_meter: UnivariateStatistic | RunningStatistic,
        average_meter: UnivariateStatistic | RunningStatistic,
    ) -> None:
        self.variance_meter = variance_meter
        self.average_meter = average_meter
        self.num_seen_ = 0

    def fit_partial(self, score: float) -> _ZScorePostprocessor:
        """Fits the postprocessor to the (next) timestep's score. Running postprocessors only keep the scores in their window.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.variance_meter.update(score)
        self.average_meter.update(score)
        self.num_seen_ += 1

        return self

    def transform_partial(self, score: float) -> float:
        """Applies postprocessing to the score, using the statistics of the window for running postprocessors.

        Args:
            score (float): The input score.

        Returns:
            float: Transformed score, or 0.0 before the first fit.
        """
        if self.num_seen_ == 0:
            return 0.0
        variance = self.variance_meter.get()
        # Scores in a zero-variance window equal the mean, so their normalized
        # deviation is zero. Returning 0.0 also keeps downstream ensemblers usable.
        if variance == 0:
            return 0.0

        zscore = (score - self.average_meter.get()) / np.sqrt(variance)

        return zscore

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Normalizes a batch of scores using the current frozen mean and variance."""
        if isinstance(scores, np.ndarray) and scores.ndim == 1:
            n = len(scores)
            if n == 0 or self.num_seen_ == 0:
                return np.zeros(n, dtype=np.float64)
            variance = self.variance_meter.get()
            if variance == 0:
                return np.zeros(n, dtype=np.float64)
            mean = self.average_meter.get()
            return np.asarray((scores - mean) / np.sqrt(variance), dtype=np.float64)
        return super().transform(scores)


class AveragePostprocessor(_MeterPostprocessor):
    """A postprocessor that converts a score to the average of all previous scores."""

    def __init__(self) -> None:
        super().__init__(AverageMeter())

    def fit(self, scores: np.ndarray) -> AveragePostprocessor:
        if isinstance(scores, np.ndarray) and scores.ndim == 1 and not np.isnan(scores).any():
            if len(scores) > 0:
                meter = cast(AverageMeter, self.meter)
                meter.sum += float(scores.astype(np.float64, copy=False).sum())
                meter.num_items += len(scores)
                self.num_seen_ += len(scores)
            return self
        return cast(AveragePostprocessor, super().fit(scores))

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        if isinstance(scores, np.ndarray) and scores.ndim == 1 and not np.isnan(scores).any():
            if len(scores) == 0:
                return np.empty(0, dtype=np.float64)
            arr = scores.astype(np.float64, copy=False)
            meter = cast(AverageMeter, self.meter)
            cumsum = meter.sum + arr.cumsum()
            counts = np.arange(
                meter.num_items + 1, meter.num_items + len(arr) + 1, dtype=np.float64
            )
            meter.sum = float(cumsum[-1])
            meter.num_items += len(arr)
            self.num_seen_ += len(arr)
            return cumsum / counts
        return super().fit_transform(scores)


class MaxPostprocessor(_MeterPostprocessor):
    """A postprocessor that converts a score to the maximum of all previous scores."""

    def __init__(self) -> None:
        # Nothing is ever removed from the whole stream, so the maximum alone is enough (#388).
        super().__init__(MaxMeter(keep_values=False))

    def fit(self, scores: np.ndarray) -> MaxPostprocessor:
        if isinstance(scores, np.ndarray) and scores.ndim == 1 and not np.isnan(scores).any():
            if len(scores) > 0:
                meter = cast(MaxMeter, self.meter)
                batch_max = float(scores.max())
                if batch_max > meter.max:
                    meter.max = batch_max
                self.num_seen_ += len(scores)
            return self
        return cast(MaxPostprocessor, super().fit(scores))

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        if isinstance(scores, np.ndarray) and scores.ndim == 1 and not np.isnan(scores).any():
            if len(scores) == 0:
                return np.empty(0, dtype=np.float64)
            arr = scores.astype(np.float64, copy=False)
            meter = cast(MaxMeter, self.meter)
            acc = np.maximum(meter.max, np.maximum.accumulate(arr))
            meter.max = float(acc[-1])
            self.num_seen_ += len(arr)
            return acc
        return super().fit_transform(scores)


class ZScorePostprocessor(_ZScorePostprocessor):
    """A postprocessor that normalizes the score via Z-score normalization."""

    def __init__(self) -> None:
        super().__init__(VarianceMeter(), AverageMeter())


class QuantileFilter(BasePostprocessor):
    """A postprocessor that labels a score 1 if it is at least the ``q`` quantile of the previous scores, and 0 otherwise.

    This is the ``QuantileFilter`` of River (https://github.com/online-ml/river, BSD-3-Clause license). The quantile comes from :class:`~pysad.statistics.QuantileMeter`, the P^2 algorithm of :cite:`jain1985psquare`, so the memory stays constant.
    ``fit_transform_partial`` labels the score with the quantile of the scores before it, then fits the score, as River does.

    These are the differences from River:

    - River labels a score 0 when the quantile of the previous scores is exactly 0. This class labels it 1 if the score is at least 0.
    - River does not learn the detector from the scores that it labels 1, unless ``protect_anomaly_detector=False``. A postprocessor does not own the detector, so this class leaves that choice to the caller, see :doc:`/guides/alerts`.
    - With 2 to 5 previous scores, the quantile can differ from River's. :class:`~pysad.statistics.QuantileMeter` interpolates between the sorted scores for the first four scores and uses the central P^2 marker from the fifth score, as the paper does. River's ``Quantile`` takes one of the sorted scores until the sixth score. With more previous scores, the two quantiles and the labels are the same.

    Args:
        q (float): The quantile of the previous scores that a score must reach to get the label 1, in the open interval (0, 1) (Default=0.95).
    """

    def __init__(self, q: float = 0.95) -> None:
        self.meter = QuantileMeter(q=q)

    def fit_partial(self, score: float) -> QuantileFilter:
        """Updates the quantile with the next score.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.meter.update(score)

        return self

    def transform_partial(self, score: float) -> float:
        """Labels the score with the quantile of the fitted scores.

        Args:
            score (float): The input score.

        Returns:
            float: 1.0 if the score is at least the quantile of the fitted scores, and 0.0 otherwise or if no score was fitted.
        """
        return float(self.meter.num_items > 0 and score >= self.meter.get())

    def fit_transform_partial(self, score: float) -> float:
        """Labels the score with the quantile of the previous scores, then fits the score.

        Args:
            score (float): Input score.

        Returns:
            float: 1.0 if the score is at least the quantile of the previous scores, and 0.0 otherwise.
        """
        label = self.transform_partial(score)
        self.fit_partial(score)
        return label

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Labels a batch of scores against the current frozen quantile."""
        arr = np.asarray(scores)
        if arr.ndim == 1:
            if len(arr) == 0 or self.meter.num_items == 0:
                return np.zeros(len(arr), dtype=np.float64)
            q_val = self.meter.get()
            return (arr >= q_val).astype(np.float64)
        return super().transform(scores)


class ThresholdFilter(BasePostprocessor):
    """A postprocessor that turns a score into a label with a fixed threshold, like ``ThresholdFilter`` of River.

    The output is 1.0 (anomalous) if the score is at least ``threshold``, and 0.0 (normal) otherwise. Unlike River, the filter does not wrap a detector and does not keep the anomalies out of the detector. To keep them out, call ``fit_partial`` of the model only for the points with the output 0.

    Args:
        threshold (float): The lowest score that gets the label 1.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def fit_partial(self, score: float) -> ThresholdFilter:
        """Does nothing, because the threshold is fixed.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        return self

    def transform_partial(self, score: float) -> float:
        """Labels the score.

        Args:
            score (float): The input score.

        Returns:
            float: 1.0 if the score is at least ``threshold``, 0.0 otherwise.
        """
        return float(score >= self.threshold)

    def fit(self, scores: np.ndarray) -> ThresholdFilter:
        """Does nothing, because the threshold is fixed."""
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Labels all scores in a 1-D batch at once."""
        arr = np.asarray(scores)
        if arr.ndim == 1:
            return (arr >= self.threshold).astype(np.float64)
        return super().transform(scores)

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        """Labels all scores in a 1-D batch at once."""
        return self.transform(scores)
