from pysad.core.base_postprocessor import BasePostprocessor
from pysad.statistics.average_meter import AverageMeter
from pysad.statistics.max_meter import MaxMeter
from pysad.statistics.median_meter import MedianMeter
from pysad.statistics.running_statistic import RunningStatistic
from pysad.statistics.variance_meter import VarianceMeter
import numpy as np


class RunningAveragePostprocessor(BasePostprocessor):
    """A postprocessor that convert a score to the average of of all previous scores in the window.

        Args:
            window_size (int): Length of the window
    """

    def __init__(self, window_size):

        self.meter = RunningStatistic(
            statistic_cls=AverageMeter,
            window_size=window_size)

    def fit_partial(self, score):
        """Fits the windowed postprocessor to the (next) timestep's score.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.meter.update(score)

        return self

    def transform_partial(self, score=None):
        """Applies postprocessing to the score using the window.

        Args:
            score (float): The input score.

        Returns:
            float: Transformed score.
        """
        return self.meter.get()


class RunningMaxPostprocessor(BasePostprocessor):
    """A postprocessor that convert a score to the maximum of of all previous scores in the window.
        Args:
            window_size (int): Length of the window
    """

    def __init__(self, window_size):

        self.meter = RunningStatistic(
            statistic_cls=MaxMeter,
            window_size=window_size)

    def fit_partial(self, score):
        """Fits the windowed postprocessor to the (next) timestep's score.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.meter.update(score)

        return self

    def transform_partial(self, score=None):
        """Applies postprocessing to the score using the window.

        Args:
            score (float): The input score.

        Returns:
            float: Transformed score.
        """
        return self.meter.get()


class RunningMedianPostprocessor(BasePostprocessor):
    """A postprocessor that convert a score to the median of of all previous scores in the window.
        Args:
            window_size (int): Length of the window
    """

    def __init__(self, window_size):

        self.meter = RunningStatistic(
            statistic_cls=MedianMeter,
            window_size=window_size)

    def fit_partial(self, score):
        """Fits the windowed postprocessor to the (next) timestep's score.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.meter.update(score)

        return self

    def transform_partial(self, score=None):
        """Applies postprocessing to the score using the window.

        Args:
            score (float): The input score.

        Returns:
            float: Transformed score.
        """
        return self.meter.get()


class RunningZScorePostprocessor(BasePostprocessor):
    """A postprocessor that normalizes score using Z-score normalization with the statistics of the window.

        Args:
            window_size (int): Length of the window
    """

    def __init__(self, window_size):

        self.variance_meter = RunningStatistic(
            statistic_cls=VarianceMeter, window_size=window_size)
        self.average_meter = RunningStatistic(
            statistic_cls=AverageMeter, window_size=window_size)

    def fit_partial(self, score):
        """Fits the windowed postprocessor to the (next) timestep's score.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.variance_meter.update(score)
        self.average_meter.update(score)

        return self

    def transform_partial(self, score):
        """Applies postprocessing to the score using the window.

        Args:
            score (float): The input score.

        Returns:
            float: Transformed score.
        """
        zscore = (score - self.average_meter.get()) / \
            np.sqrt(self.variance_meter.get())

        return zscore
