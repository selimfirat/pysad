from postprocessing.base_postprocessor import BasePostProcessor
from stats.average_meter import AverageMeter
from stats.max_meter import MaxMeter
from stats.median_meter import MedianMeter
from stats.variance_meter import VarianceMeter
import numpy as np


class AveragePostprocessor(BasePostProcessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meter = AverageMeter()

    def fit_partial(self, score):

        self.meter.update(score)

        return self

    def transform_partial(self, score=None):

        return self.meter.get()


class MaxPostprocessor(BasePostProcessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meter = MaxMeter()

    def fit_partial(self, score):
        self.meter.update(score)

        return self

    def transform_partial(self, score=None):
        return self.meter.get()


class MedianPostprocessor(BasePostProcessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meter = MedianMeter()

    def fit_partial(self, score):
        self.meter.update(score)

        return self

    def transform_partial(self, score=None):

        return self.meter.get()


class ZScorePostprocessor(BasePostProcessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.variance_meter = VarianceMeter()
        self.average_meter = AverageMeter()

    def fit_partial(self, score):
        self.variance_meter.update(score)
        self.average_meter.update(score)

        return self

    def transform_partial(self, score):

        zscore = (score - self.average_meter.get()) / np.sqrt(self.variance_meter.get())

        return zscore
