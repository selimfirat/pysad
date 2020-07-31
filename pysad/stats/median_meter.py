from heapq import heappush
from pysad.stats.base_statistic import UnivariateStatistic


class MedianMeter(UnivariateStatistic):
    """The statistic that keeps track of the median.

        Attrs:
        num_items: int
            The number of items that are used to update the statistic.
        lst: list<float>
            The list of values that are used to update the statistic.

    """
    def __init__(self):
        self.lst = []
        self.num_items = 0

    def update(self, num):
        """Updates the statistic with the value for a timestep.

        Args:
            num: The incoming value, for which the statistic is used.

        Returns:
            self: object
                Returns the fitted statistic.

        """
        heappush(self.lst, num)
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
        self.lst.remove(num)
        self.num_items -= 1


        return self

    def get(self):
        """ Method to obtain the tracked statistic.

        Returns:
            statistic: float
                The statistic.
        """
        if self.num_items % 2 == 0:
            return (self.lst[self.num_items // 2] + self.lst[self.num_items//2 - 1]) / 2
        else:
            return self.lst[self.num_items//2]
