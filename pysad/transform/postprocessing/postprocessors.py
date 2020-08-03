from pysad.core.base_transformer import BaseTransformer
from pysad.stats.average_meter import AverageMeter
from pysad.stats.max_meter import MaxMeter
from pysad.stats.median_meter import MedianMeter
from pysad.stats.variance_meter import VarianceMeter
import numpy as np


class AveragePostprocessor(BaseTransformer):
    """A postprocessor that convert a score to the average of of all previous scores.
    """
    def __init__(self):
        self.meter = AverageMeter()

    def fit_partial(self, score):
        """Fits the postprocessor to the (next) timestep's score.

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
        """Gets the current average. This method should be used immediately after the fit_partial method with same score.

        Args:
            score: float
                The input score.

        Returns:
            result_score: float
                Transformed score.
        """
        return self.meter.get()


class MaxPostprocessor(BaseTransformer):
    """A postprocessor that convert a score to the maximum of of all previous scores.
    """

    def __init__(self):
        self.meter = MaxMeter()

    def fit_partial(self, score):
        """Fits the postprocessor to the (next) timestep's score.

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
        """Applies postprocessing to the score. This method should be used immediately after the fit_partial method with same score.

        Args:
            score: float
                The input score.

        Returns:
            result_score: float
                Transformed score.
        """
        return self.meter.get()


class MedianPostprocessor(BaseTransformer):
    """A postprocessor that convert a score to the median of of all previous scores.
    """

    def __init__(self):
        self.meter = MedianMeter()

    def fit_partial(self, score):
        """Fits the postprocessor to the (next) timestep's score.

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
        """Applies postprocessing to the score.

        Args:
            score: float
                The input score.

        Returns:
            result_score: float
                Transformed score.
        """
        return self.meter.get()


class ZScorePostprocessor(BaseTransformer):
    """A postprocessor that normalize the score via Z-score normalization.
    """

    def __init__(self):
        self.variance_meter = VarianceMeter()
        self.average_meter = AverageMeter()

    def fit_partial(self, score):
        """Fits the postprocessor to the (next) timestep's score.

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
        """Applies postprocessing to the score.

        Args:
            score: float
                The input score.

        Returns:
            result_score: float
                Transformed score.
        """
        zscore = (score - self.average_meter.get()) / np.sqrt(self.variance_meter.get())

        return zscore
