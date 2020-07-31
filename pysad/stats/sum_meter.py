from pysad.stats.base_statistic import UnivariateStatistic


class SumMeter(UnivariateStatistic):
    """The statistic that keeps the sum of values.

    Attrs:
        sum: float
            The summation of values.
        num_items: int
            The number of items that are used to update the statistic.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sum = 0.0
        self.num_items = 0

    def update(self, num):
        """Updates the statistic with the value for a timestep.

        Args:
            num: The incoming value, for which the statistic is used.

        Returns:
            self: object
                Returns the fitted statistic.

        """
        self.sum += num
        self.num_items += 1

        return self

    def remove(self, num):
        """Updates the statistic by removing particular value. This method

        Args:
            num: The value to be removed.

        Returns:
            self: object
                Returns the fitted statistic.

        """
        self.sum -= num
        self.num_items -= 1

        return self

    def get(self):
        """ Method to obtain the tracked statistic.

        Returns:
            statistic: float
                The statistic.
        """
        return self.sum
