from __future__ import annotations

import math
from collections import deque

from pysad.core.base_statistic import UnivariateStatistic


class MinMeter(UnivariateStatistic):
    """The statistic that keeps track of the minimum value.

    Removing the oldest value, which is what :class:`~pysad.statistics.RunningStatistic` does, takes amortized constant time. Removing any other value takes time linear in the number of values.

    Attributes:
        min (float): The minimum value.
        lst (collections.deque[float]): The values that are used to update the statistic, oldest first. It is necessary for windowing operations.
        candidates (collections.deque[float]): The values that can still become the minimum when older values are removed, oldest first. They are in non-decreasing order.
    """

    def __init__(self) -> None:
        self.min = math.inf

        self.lst: deque[float] = deque()
        self.candidates: deque[float] = deque()

    def update(self, num: float) -> MinMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        if num < self.min:
            self.min = num

        self.lst.append(num)
        self._push(num)

        return self

    def remove(self, num: float) -> MinMeter:
        """Updates the statistic by removing a particular value.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        if self.lst and self.lst[0] == num:
            self.lst.popleft()
            # The oldest value is the first candidate unless a smaller value came after it.
            if self.candidates[0] == num:
                self.candidates.popleft()
        else:
            self.lst.remove(num)
            self.candidates.clear()
            for value in self.lst:
                self._push(value)

        self.min = self.candidates[0] if self.candidates else math.inf

        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        return self.min

    def _push(self, num: float) -> None:
        # A candidate larger than num is removed before num, so it can no longer be the minimum.
        while self.candidates and self.candidates[-1] > num:
            self.candidates.pop()
        self.candidates.append(num)
