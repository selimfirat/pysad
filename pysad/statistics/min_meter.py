from __future__ import annotations

import math
from collections import deque

from pysad.core.base_statistic import UnivariateStatistic


class MinMeter(UnivariateStatistic):
    """The statistic that keeps track of the minimum value.

    Removing the oldest value, which is what :class:`~pysad.statistics.RunningStatistic` does, takes amortized constant time. Removing any other value takes time linear in the number of values.

    Args:
        keep_values (bool): Whether to keep every value, which :meth:`remove` needs (windowing).
            With ``False`` only the minimum is kept, so memory does not grow with the stream (Default=True).

    Attributes:
        min (float): The minimum value.
        lst (collections.deque[float]): The values that are used to update the statistic, oldest first. It is necessary for windowing operations.
            Stays empty when ``keep_values`` is ``False``.
        candidates (collections.deque[float]): The values that can still become the minimum when older values are removed, oldest first. They are in non-decreasing order.
            Stays empty when ``keep_values`` is ``False``.
    """

    def __init__(self, keep_values: bool = True) -> None:
        self.min = math.inf
        self.keep_values = keep_values

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

        if self.keep_values:
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
        if not self.keep_values:
            raise ValueError(
                "MinMeter(keep_values=False) keeps no values, so it cannot remove one."
            )

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
