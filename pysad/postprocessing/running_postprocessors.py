from pysad.postprocessing.base_postprocessor import BasePostProcessor
from pysad.stats.average_meter import AverageMeter
from pysad.stats.max_meter import MaxMeter
from pysad.stats.median_meter import MedianMeter
from pysad.stats.running_statistic import RunningStatistic
from pysad.stats.variance_meter import VarianceMeter
import numpy as np


class RunningAveragePostprocessor(BasePostProcessor):
    """A postprocessor that convert a score to the average of of all previous scores in the window.
        Args:
            window_size: Length of the window
    """

    def __init__(self, window_size):

        self.meter = RunningStatistic(statistic_cls=AverageMeter, window_size=window_size)

    def fit_partial(self, score):
        """Fits the windowed postprocessor to the (next) timestep's score.

        Args:
            score: float
                Input score.

        Returns:
            self: object
                Fitted postprocessor.
        """
        self.meter.update(score)

        return self

    def transform_partial(self, score=None):
        """Applies postprocessing to the score using the window.

        Args:
            score: float
                The input score.

        Returns:
            result_score: float
                Transformed score.
        """
        return self.meter.get()


class RunningMaxPostprocessor(BasePostProcessor):
    """A postprocessor that convert a score to the maximum of of all previous scores in the window.
        Args:
            window_size: Length of the window
    """

    def __init__(self, window_size):

        self.meter = RunningStatistic(statistic_cls=MaxMeter, window_size=window_size)

    def fit_partial(self, score):
        """Fits the windowed postprocessor to the (next) timestep's score.

        Args:
            score: float
                Input score.

        Returns:
            self: object
                Fitted postprocessor.
        """
        self.meter.update(score)

        return self

    def transform_partial(self, score=None):
        """Applies postprocessing to the score using the window.

        Args:
            score: float
                The input score.

        Returns:
            result_score: float
                Transformed score.
        """
        return self.meter.get()


class RunningMedianPostprocessor(BasePostProcessor):
    """A postprocessor that convert a score to the median of of all previous scores in the window.
        Args:
            window_size: Length of the window
    """

    def __init__(self, window_size):

        self.meter = RunningStatistic(statistic_cls=MedianMeter, window_size=window_size)

    def fit_partial(self, score):
        """Fits the windowed postprocessor to the (next) timestep's score.

        Args:
            score: float
                Input score.

        Returns:
            self: object
                Fitted postprocessor.
        """
        self.meter.update(score)

        return self

    def transform_partial(self, score=None):
        """Applies postprocessing to the score using the window.

        Args:
            score: float
                The input score.

        Returns:
            result_score: float
                Transformed score.
        """
        return self.meter.get()


class RunningZScorePostprocessor(BasePostProcessor):
    """A postprocessor that normalizes score using Z-score normalization with the statistics of the window.
        Args:
            window_size: Length of the window
    """

    def __init__(self, window_size):

        self.variance_meter = RunningStatistic(statistic_cls=VarianceMeter, window_size=window_size)
        self.average_meter = RunningStatistic(statistic_cls=AverageMeter, window_size=window_size)

    def fit_partial(self, score):
        """Fits the windowed postprocessor to the (next) timestep's score.

        Args:
            score: float
                Input score.

        Returns:
            self: object
                Fitted postprocessor.
        """
        self.variance_meter.update(score)
        self.average_meter.update(score)

        return self

    def transform_partial(self, score):
        """Applies postprocessing to the score using the window.

        Args:
            score: float
                The input score.

        Returns:
            result_score: float
                Transformed score.
        """
        zscore = (score - self.average_meter.get()) / np.sqrt(self.variance_meter.get())

        return zscore
