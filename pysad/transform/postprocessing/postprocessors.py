from __future__ import annotations

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

    def fit_partial(self, score: float) -> _MeterPostprocessor:
        """Fits the postprocessor to the (next) timestep's score. Running postprocessors only keep the scores in their window.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.meter.update(score)

        return self

    def transform_partial(self, score: float | None = None) -> float:
        """Applies postprocessing to the score, using only the scores in the window for running postprocessors. This method should be used immediately after the fit_partial method with same score.

        Args:
            score (float): The input score.

        Returns:
            float: Transformed score.
        """
        return self.meter.get()


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

    def fit_partial(self, score: float) -> _ZScorePostprocessor:
        """Fits the postprocessor to the (next) timestep's score. Running postprocessors only keep the scores in their window.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.variance_meter.update(score)
        self.average_meter.update(score)

        return self

    def transform_partial(self, score: float) -> float:
        """Applies postprocessing to the score, using the statistics of the window for running postprocessors.

        Args:
            score (float): The input score.

        Returns:
            float: Transformed score.
        """
        variance = self.variance_meter.get()
        # Scores in a zero-variance window equal the mean, so their normalized
        # deviation is zero. Returning 0.0 also keeps downstream ensemblers usable.
        if variance == 0:
            return 0.0

        zscore = (score - self.average_meter.get()) / np.sqrt(variance)

        return zscore


class AveragePostprocessor(_MeterPostprocessor):
    """A postprocessor that converts a score to the average of all previous scores."""

    def __init__(self) -> None:
        super().__init__(AverageMeter())


class MaxPostprocessor(_MeterPostprocessor):
    """A postprocessor that converts a score to the maximum of all previous scores."""

    def __init__(self) -> None:
        # Nothing is ever removed from the whole stream, so the maximum alone is enough (#388).
        super().__init__(MaxMeter(keep_values=False))


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
