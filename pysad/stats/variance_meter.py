from pysad.stats.base_statistic import UnivariateStatistic
from pysad.stats.count_meter import CountMeter
from pysad.stats.sum_meter import SumMeter
from pysad.stats.sum_squares_meter import SumSquaresMeter


class VarianceMeter(UnivariateStatistic):
    """The statistic that keeps track of the variance of the values.

    Attrs:
        sum_meter: SumMeter object
        sum_squares_meter: SumSquaresMeter object
        count_meter: CountMeter
    """

    def __init__(self):
        self.sum_meter = SumMeter()
        self.sum_squares_meter = SumSquaresMeter()
        self.count_meter = CountMeter()

    def update(self, num):
        """Updates the statistic with the value for a timestep.

        Args:
            num: The incoming value, for which the statistic is used.

        Returns:
            self: object
                Returns the fitted statistic.

        """
        self.sum_squares_meter.update(num)
        self.count_meter.update(num)
        self.sum_meter.update(num)

        return self

    def remove(self, num):
        """Updates the statistic by removing particular value. This method

        Args:
            num: The value to be removed.

        Returns:
            self: object
                Returns the fitted statistic.

        """
        self.sum_squares_meter.remove(num)
        self.sum_meter.remove(num)
        self.count_meter.remove(num)

        return self

    def get(self):
        """ Method to obtain the tracked statistic.

        Returns:
            statistic: float
                The statistic.
        """
        sum_squares = self.sum_squares_meter.get()
        sum = self.sum_meter.get()
        count = self.count_meter.get()

        var = (sum_squares - (sum**2)/count)/(count - 1)

        return var
