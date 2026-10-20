from __future__ import annotations

from bisect import bisect_left, insort

from pysad.core.base_statistic import UnivariateStatistic


class MedianMeter(UnivariateStatistic):
    """The statistic that keeps track of the median.

    Attributes:
        num_items (int): The number of items that are used to update the statistic.
        lst (list[float]): The list of values that are used to update the statistic. It is necessary for windowing operations.
    """

    def __init__(self) -> None:
        self.lst: list[float] = []
        self.num_items = 0

    def update(self, num: float) -> MedianMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        insort(self.lst, num)
        self.num_items += 1

        return self

    def remove(self, num: float) -> MedianMeter:
        """Updates the statistic by removing a particular value.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        idx = bisect_left(self.lst, num)
        if idx < self.num_items and self.lst[idx] == num:
            del self.lst[idx]
            self.num_items -= 1
        else:
            raise ValueError(f"{num} not in MedianMeter")

        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        if self.num_items % 2 == 0:
            return (self.lst[self.num_items // 2] + self.lst[self.num_items // 2 - 1]) / 2
        else:
            return self.lst[self.num_items // 2]
