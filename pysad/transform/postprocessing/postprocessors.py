from pysad.core.base_postprocessor import BasePostprocessor
from pysad.statistics.average_meter import AverageMeter
from pysad.statistics.max_meter import MaxMeter
from pysad.statistics.median_meter import MedianMeter
from pysad.statistics.variance_meter import VarianceMeter
import numpy as np


class _MeterPostprocessor(BasePostprocessor):
    """Base class for postprocessors that convert a score to a statistic of the previous scores.

        Args:
            meter: The statistic to update with each score.
    """

    def __init__(self, meter):
        self.meter = meter

    def fit_partial(self, score):
        """Fits the postprocessor to the (next) timestep's score. Running postprocessors only keep the scores in their window.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.meter.update(score)

        return self

    def transform_partial(self, score=None):
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

    def __init__(self, variance_meter, average_meter):
        self.variance_meter = variance_meter
        self.average_meter = average_meter

    def fit_partial(self, score):
        """Fits the postprocessor to the (next) timestep's score. Running postprocessors only keep the scores in their window.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.variance_meter.update(score)
        self.average_meter.update(score)

        return self

    def transform_partial(self, score):
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
    """A postprocessor that convert a score to the average of of all previous scores.
    """

    def __init__(self):
        super().__init__(AverageMeter())


class MaxPostprocessor(_MeterPostprocessor):
    """A postprocessor that convert a score to the maximum of of all previous scores.
    """

    def __init__(self):
        super().__init__(MaxMeter())


class MedianPostprocessor(_MeterPostprocessor):
    """A postprocessor that convert a score to the median of of all previous scores.
    """

    def __init__(self):
        super().__init__(MedianMeter())


class ZScorePostprocessor(_ZScorePostprocessor):
    """A postprocessor that normalize the score via Z-score normalization.
    """

    def __init__(self):
        super().__init__(VarianceMeter(), AverageMeter())
