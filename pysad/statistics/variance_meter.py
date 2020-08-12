from pysad.core.base_statistic import UnivariateStatistic
from pysad.statistics.count_meter import CountMeter
from pysad.statistics.sum_meter import SumMeter
from pysad.statistics.sum_squares_meter import SumSquaresMeter


class VarianceMeter(UnivariateStatistic):
    """The statistic that keeps track of the variance of the values. The variance formula is: (sum_squares - (sum**2)/count)/count.

    Attrs:
        sum_meter (pyod.statistics.SumMeter object): SumMeter object.
        sum_squares_meter (pyod.statistics.SumSquaresMeter object): SumSquaresMeter object.
        count_meter (pyod.statistics.CountMeter object): CountMeter object.
    """

    def __init__(self):
        self.sum_meter = SumMeter()
        self.sum_squares_meter = SumSquaresMeter()
        self.count_meter = CountMeter()

    def update(self, num):
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.

        """
        self.sum_squares_meter.update(num)
        self.count_meter.update(num)
        self.sum_meter.update(num)

        return self

    def remove(self, num):
        """Updates the statistic by removing particular value.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.

        """
        self.sum_squares_meter.remove(num)
        self.sum_meter.remove(num)
        self.count_meter.remove(num)

        return self

    def get(self):
        """ Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        sum_squares = self.sum_squares_meter.get()
        sum = self.sum_meter.get()
        count = self.count_meter.get()

        var = (sum_squares - (sum**2) / count) / count

        return var
