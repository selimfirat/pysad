from postprocessing.base_postprocessor import BasePostProcessor
from stats.average_meter import AverageMeter
from stats.max_meter import MaxMeter
from stats.median_meter import MedianMeter
from stats.running_statistic import RunningStatistic
from stats.variance_meter import VarianceMeter
import numpy as np


class RunningAveragePostprocessor(BasePostProcessor):

    def __init__(self, window_size, **kwargs):
        super().__init__(**kwargs)

        self.meter = RunningStatistic(statistic_cls=AverageMeter, window_size=window_size)

    def fit_partial(self, score):

        self.meter.update(score)

        return self

    def transform_partial(self, score=None):

        return self.meter.get()


class RunningMaxPostprocessor(BasePostProcessor):

    def __init__(self, window_size, **kwargs):
        super().__init__(**kwargs)

        self.meter = RunningStatistic(statistic_cls=MaxMeter, window_size=window_size)

    def fit_partial(self, score):
        self.meter.update(score)

        return self

    def transform_partial(self, score=None):
        return self.meter.get()


class RunningMedianPostprocessor(BasePostProcessor):

    def __init__(self, window_size, **kwargs):
        super().__init__(**kwargs)

        self.meter = RunningStatistic(statistic_cls=MedianMeter, window_size=window_size)

    def fit_partial(self, score):
        self.meter.update(score)

        return self

    def transform_partial(self, score=None):

        return self.meter.get()


class RunningZScorePostprocessor(BasePostProcessor):

    def __init__(self, window_size, **kwargs):
        super().__init__(**kwargs)

        self.variance_meter = RunningStatistic(statistic_cls=VarianceMeter, window_size=window_size)
        self.average_meter = RunningStatistic(statistic_cls=AverageMeter, window_size=window_size)

    def fit_partial(self, score):
        self.variance_meter.update(score)
        self.average_meter.update(score)

        return self

    def transform_partial(self, score):

        zscore = (score - self.average_meter.get()) / np.sqrt(self.variance_meter.get())

        return zscore
