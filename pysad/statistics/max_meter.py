import math
from heapq import heappush
from pysad.core.base_statistic import UnivariateStatistic
import numpy as np


class MaxMeter(UnivariateStatistic):
    """The statistic that keeps track of the maximum value.

        Attrs:
            max (float): The maximum value.
            lst (list[float]): The list of values that are used to update the statistic. It is necessary for windowing operations.
    """

    def __init__(self):
        self.max = -math.inf

        self.lst = []

    def update(self, num):
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.

        """
        if num > self.max:
            self.max = num

        heappush(self.lst, num)

        return self

    def remove(self, num):
        """Updates the statistic by removing particular value. This method

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.

        """
        self.lst.remove(num)

        if len(self.lst) > 0:
            self.max = np.max(self.lst)
        else:
            self.max = -math.inf

        return self

    def get(self):
        """ Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        return self.max
