from pysad.stats.count_meter import CountMeter
from pysad.stats.sum_meter import SumMeter
from pysad.stats.sum_squares_meter import SumSquaresMeter
from pysad.stats.univariate_statistic import UnivariateStatistic


class VarianceMeter(UnivariateStatistic):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sum_meter = SumMeter()
        self.sum_squares_meter = SumSquaresMeter()
        self.count_meter = CountMeter()

    def update(self, num):
        self.sum_squares_meter.update(num)
        self.count_meter.update(num)
        self.sum_meter.update(num)

        return self

    def remove(self, num):

        self.sum_squares_meter.remove(num)
        self.sum_meter.remove(num)
        self.count_meter.remove(num)

        return self

    def get(self):
        sum_squares = self.sum_squares_meter.get()
        sum = self.sum_meter.get()
        count = self.count_meter.get()

        var = (sum_squares - (sum**2)/count)/(count - 1)

        return var
