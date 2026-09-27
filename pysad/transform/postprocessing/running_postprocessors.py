from pysad.statistics.average_meter import AverageMeter
from pysad.statistics.max_meter import MaxMeter
from pysad.statistics.median_meter import MedianMeter
from pysad.statistics.running_statistic import RunningStatistic
from pysad.statistics.variance_meter import VarianceMeter
from pysad.transform.postprocessing.postprocessors import _MeterPostprocessor, _ZScorePostprocessor


class RunningAveragePostprocessor(_MeterPostprocessor):
    """A postprocessor that convert a score to the average of of all previous scores in the window.

        Args:
            window_size (int): Length of the window
    """

    def __init__(self, window_size):
        super().__init__(RunningStatistic(statistic_cls=AverageMeter, window_size=window_size))


class RunningMaxPostprocessor(_MeterPostprocessor):
    """A postprocessor that convert a score to the maximum of of all previous scores in the window.
        Args:
            window_size (int): Length of the window
    """

    def __init__(self, window_size):
        super().__init__(RunningStatistic(statistic_cls=MaxMeter, window_size=window_size))


class RunningMedianPostprocessor(_MeterPostprocessor):
    """A postprocessor that convert a score to the median of of all previous scores in the window.
        Args:
            window_size (int): Length of the window
    """

    def __init__(self, window_size):
        super().__init__(RunningStatistic(statistic_cls=MedianMeter, window_size=window_size))


class RunningZScorePostprocessor(_ZScorePostprocessor):
    """A postprocessor that normalizes score using Z-score normalization with the statistics of the window.

        Args:
            window_size (int): Length of the window
    """

    def __init__(self, window_size):
        super().__init__(
            RunningStatistic(statistic_cls=VarianceMeter, window_size=window_size),
            RunningStatistic(statistic_cls=AverageMeter, window_size=window_size))
