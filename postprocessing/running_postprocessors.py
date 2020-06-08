from stats.average_meter import AverageMeter
from stats.max_meter import MaxMeter
from stats.median_meter import MedianMeter
from stats.running_statistic import RunningStatistic


class RunningAveragePostprocessor(RunningStatistic):

    def __init__(self, window_size, **kwargs):
        super().__init__(AverageMeter, window_size, **kwargs)


class RunningMaxPostprocessor(RunningStatistic):

    def __init__(self, window_size, **kwargs):
        super().__init__(MaxMeter, window_size, **kwargs)


class RunningMedianPostprocessor(RunningStatistic):

    def __init__(self, window_size, **kwargs):
        super().__init__(MedianMeter, window_size, **kwargs)
