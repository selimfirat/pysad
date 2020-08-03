from pysad.core.base_model import BaseModel
from pysad.stats.average_meter import AverageMeter
from pysad.stats.median_meter import MedianMeter
from pysad.stats.variance_meter import VarianceMeter


class StandardAbsoluteDeviation(BaseModel):

    def __init__(self, substracted_statistic="mean", absolute=True):
        """
        3-Sigma rule described in https://arxiv.org/pdf/1704.07706.pdf
        :param kwargs:
        """

        self.absolute = absolute
        self.variance_meter = VarianceMeter()

        if substracted_statistic == "median":
            self.sub_meter = MedianMeter()
        elif substracted_statistic == "mean":
            self.sub_meter = AverageMeter()
        else:
            raise ValueError("Unknown substracted_statistic value! Please choose median or mean.")

    def fit_partial(self, x, y=None):
        assert len(x) == 1 # Only for time series

        self.variance_meter.update(x)
        self.sub_meter.update(x)

    def score_partial(self, x):
        sub = self.sub_meter.get()
        dev = self.variance_meter.get()**0.5

        score = (x - sub) / dev

        return abs(score) if self.absolute else score
