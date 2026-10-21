from __future__ import annotations

import math
from heapq import heappush

import numpy as np

from pysad.core.base_statistic import UnivariateStatistic


class MaxMeter(UnivariateStatistic):
    """The statistic that keeps track of the maximum value.

    Args:
        keep_values (bool): Whether to keep every value, which :meth:`remove` needs (windowing).
            With ``False`` only the maximum is kept, so memory does not grow with the stream.

    Attributes:
        max (float): The maximum value.
        lst (list[float]): The list of values that are used to update the statistic. It is necessary for windowing operations.
            Stays empty when ``keep_values`` is ``False``.
    """

    def __init__(self, keep_values: bool = True) -> None:
        self.max = -math.inf
        self.keep_values = keep_values

        self.lst: list[float] = []

    def update(self, num: float) -> MaxMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.

        """
        if num > self.max:
            self.max = num

        if self.keep_values:
            heappush(self.lst, num)

        return self

    def remove(self, num: float) -> MaxMeter:
        """Updates the statistic by removing a particular value.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.

        """
        if not self.keep_values:
            raise ValueError(
                "MaxMeter(keep_values=False) keeps no values, so it cannot remove one."
            )

        self.lst.remove(num)

        if len(self.lst) > 0:
            self.max = np.max(self.lst)
        else:
            self.max = -math.inf

        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        return self.max
