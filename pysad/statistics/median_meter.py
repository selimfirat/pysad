from heapq import heappush
from pysad.core.base_statistic import UnivariateStatistic


class MedianMeter(UnivariateStatistic):
    """The statistic that keeps track of the median.

        Attrs:
            num_items (int): The number of items that are used to update the statistic.
            lst (list[float]): The list of values that are used to update the statistic. It is necessary for windowing operations.
    """

    def __init__(self):
        self.lst = []
        self.num_items = 0

    def update(self, num):
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        heappush(self.lst, num)
        self.num_items += 1

        return self

    def remove(self, num):
        """Updates the statistic by removing particular value. This method

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        self.lst.remove(num)
        self.num_items -= 1

        return self

    def get(self):
        """ Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        self.lst = sorted(self.lst)
        if self.num_items % 2 == 0:
            return (self.lst[self.num_items // 2] + self.lst[self.num_items // 2 - 1]) / 2
        else:
            return self.lst[self.num_items // 2]
