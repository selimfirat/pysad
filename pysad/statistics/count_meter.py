from pysad.core.base_statistic import UnivariateStatistic


class CountMeter(UnivariateStatistic):
    """A simple counter statistic.

    Attrs:
        count (int): The number of items that are used to update the statistic.
    """

    def __init__(self):
        self.count = 0

    def update(self, num):
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        self.count += 1

        return self

    def remove(self, num=None):
        """Updates the statistic by removing particular value. This method

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        self.count -= 1

        return self

    def get(self):
        """ Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        return self.count
