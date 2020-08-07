import math
from heapq import heappush
from pysad.core.base_statistic import UnivariateStatistic
import numpy as np


class MinMeter(UnivariateStatistic):
    """The statistic that keeps track of the minimum value.

        Attrs:
            min: float
                The minimum value
            lst: list<float>
                The list of values that are used to update the statistic. It is necessary for windowing operations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.min = math.inf

        self.lst = []

    def update(self, num):
        """Updates the statistic with the value for a timestep.

        Args:
            num: The incoming value, for which the statistic is used.

        Returns:
            self: object
                Returns the fitted statistic.

        """
        if num < self.min:
            self.min = num

        heappush(self.lst, num)

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

        if len(self.lst) > 0:
            self.min = np.min(self.lst)
        else:
            self.min = math.inf

        return self

    def get(self):
        """ Method to obtain the tracked statistic.

        Returns:
            statistic: float
                The statistic.
        """
        return self.min
