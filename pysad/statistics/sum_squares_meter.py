from pysad.core.base_statistic import UnivariateStatistic


class SumSquaresMeter(UnivariateStatistic):
    """The statistic that keeps track of the sum of squares. This class is useful to calculate and used in VarianceMeter.

    Attrs:
        sum_squares (float): The sum of squares of values.
        num_items (int): The number of items that are used to update the statistic.
    """

    def __init__(self):
        self.sum_squares = 0.0
        self.num_items = 0

    def update(self, num):
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        self.sum_squares += num**2
        self.num_items += 1

        return self

    def remove(self, num):
        """Updates the statistic by removing particular value. This method

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        self.sum_squares -= num**2
        self.num_items -= 1

        return self

    def get(self):
        """ Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        return self.sum_squares
