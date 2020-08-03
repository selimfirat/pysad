from pysad.core.base_model import BaseModel
from pysad.stats.median_meter import MedianMeter


class MedianAbsoluteDeviation(BaseModel):

    def __init__(self, absolute=True, b=1.4826):
        """
        Mean Absolute deviation described in https://en.wikipedia.org/wiki/Median_absolute_deviation and https://arxiv.org/pdf/1704.07706.pdf
        See https://arxiv.org/pdf/1704.07706.pdf for b parameter selection. b does not affect scores.
        :param kwargs:
        """
        super().__init__()

        self.b = b
        self.absolute = absolute
        self.median_meter = MedianMeter()
        self.mad_meter = MedianMeter()

    def fit_partial(self, x, y=None):
        assert len(x) == 1 # Only for time series

        self.median_meter.update(x)
        self.mad_meter.update(x)

    def score_partial(self, x):

        median = self.median_meter.get()
        mad = self.b*self.mad_meter.get()
        score = (x - median)/mad

        return abs(score) if self.absolute else score
