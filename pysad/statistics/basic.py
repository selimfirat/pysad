from __future__ import annotations

import math
from collections import deque
from typing import Any, cast

from pysad.core.base_statistic import BaseStatistic, RemovableStatistic


class AbsStatistic(BaseStatistic):
    """The absolute value of the statistic that is tracked.

    Args:
        statistic_cls (class): The class of the statistic to be instantiated.
        **kwargs (Keyword arguments): The keyword arguments that are passed to ``statistic_cls``.
    """

    def __init__(self, statistic_cls: type[BaseStatistic], **kwargs: Any) -> None:
        self.statistic_cls = statistic_cls
        self.statistic = self.statistic_cls(**kwargs)
        self._removable = bool(
            getattr(
                self.statistic,
                "_removable",
                callable(getattr(self.statistic, "remove", None)),
            )
        )

    def update(self, num: float) -> AbsStatistic:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        self.statistic.update(num)
        return self

    def remove(self, num: float) -> AbsStatistic:
        """Updates the statistic by removing a particular value when supported by the inner meter.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        if not self._removable:
            raise TypeError(
                f"{self.statistic_cls.__name__} does not support removing observations."
            )
        cast(RemovableStatistic, self.statistic).remove(num)
        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        return abs(self.statistic.get())


class AverageMeter(RemovableStatistic):
    """The average of the values.

    Attributes:
        sum (float): The summation of values.
        num_items (int): The number of items that are used to update the statistic.
    """

    def __init__(self) -> None:
        self.sum = 0.0
        self.num_items = 0

    def update(self, num: float) -> AverageMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        self.sum += num
        self.num_items += 1
        return self

    def remove(self, num: float) -> AverageMeter:
        """Updates the statistic by removing a particular value.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        self.sum -= num
        self.num_items -= 1
        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        return self.sum / self.num_items


class CountMeter(RemovableStatistic):
    """A simple counter statistic.

    Attributes:
        count (int): The number of items that are used to update the statistic.
    """

    def __init__(self) -> None:
        self.count = 0

    def update(self, num: float) -> CountMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        self.count += 1
        return self

    def remove(self, num: float | None = None) -> CountMeter:
        """Updates the statistic by removing a particular value.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        self.count -= 1
        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        return self.count


class MaxMeter(RemovableStatistic):
    """The statistic that keeps track of the maximum value.

    Removing the oldest value, which is what :class:`~pysad.statistics.RunningStatistic` does, takes amortized constant time. Removing any other value takes time linear in the number of values.

    Args:
        keep_values (bool): Whether to keep values needed by :meth:`remove` (windowing).
            With ``False``, only the running maximum is kept in O(1) memory; :class:`~pysad.statistics.RunningStatistic` passes ``keep_values=True`` automatically (Default=False).

    Attributes:
        max (float): The maximum value.
        lst (collections.deque[float]): The values that are used to update the statistic, oldest first. It is necessary for windowing operations.
            Stays empty when ``keep_values`` is ``False``.
        candidates (collections.deque[float]): The values that can still become the maximum when older values are removed, oldest first. They are in non-increasing order.
            Stays empty when ``keep_values`` is ``False``.
    """

    def __init__(self, keep_values: bool = False) -> None:
        self.max = -math.inf
        self.keep_values = keep_values
        self._removable = keep_values

        self.lst: deque[float] = deque()
        self.candidates: deque[float] = deque()

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
            self.lst.append(num)
            self._push(num)

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

        if self.lst and self.lst[0] == num:
            self.lst.popleft()
            # The oldest value is the first candidate unless a larger value came after it.
            if self.candidates[0] == num:
                self.candidates.popleft()
        else:
            self.lst.remove(num)
            self.candidates.clear()
            for value in self.lst:
                self._push(value)

        self.max = self.candidates[0] if self.candidates else -math.inf

        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        return self.max

    def _push(self, num: float) -> None:
        # A candidate smaller than num is removed before num, so it can no longer be the maximum.
        while self.candidates and self.candidates[-1] < num:
            self.candidates.pop()
        self.candidates.append(num)


class MinMeter(RemovableStatistic):
    """The statistic that keeps track of the minimum value.

    Removing the oldest value, which is what :class:`~pysad.statistics.RunningStatistic` does, takes amortized constant time. Removing any other value takes time linear in the number of values.

    Args:
        keep_values (bool): Whether to keep values needed by :meth:`remove` (windowing).
            With ``False``, only the running minimum is kept in O(1) memory; :class:`~pysad.statistics.RunningStatistic` passes ``keep_values=True`` automatically (Default=False).

    Attributes:
        min (float): The minimum value.
        lst (collections.deque[float]): The values that are used to update the statistic, oldest first. It is necessary for windowing operations.
            Stays empty when ``keep_values`` is ``False``.
        candidates (collections.deque[float]): The values that can still become the minimum when older values are removed, oldest first. They are in non-decreasing order.
            Stays empty when ``keep_values`` is ``False``.
    """

    def __init__(self, keep_values: bool = False) -> None:
        self.min = math.inf
        self.keep_values = keep_values
        self._removable = keep_values

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


class SumMeter(RemovableStatistic):
    """The statistic that keeps track of the sum of values.

    Attributes:
        sum (float): The summation of values.
        num_items (int): The number of items that are used to update the statistic.
    """

    def __init__(self) -> None:
        self.sum = 0.0
        self.num_items = 0

    def update(self, num: float) -> SumMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        self.sum += num
        self.num_items += 1
        return self

    def remove(self, num: float) -> SumMeter:
        """Updates the statistic by removing a particular value.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        self.sum -= num
        self.num_items -= 1
        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        return self.sum


class SumSquaresMeter(RemovableStatistic):
    """The statistic that keeps track of the sum of squares. This class is useful to calculate and used in VarianceMeter.

    Attributes:
        sum_squares (float): The sum of squares of values.
        num_items (int): The number of items that are used to update the statistic.
    """

    def __init__(self) -> None:
        self.sum_squares = 0.0
        self.num_items = 0

    def update(self, num: float) -> SumSquaresMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        self.sum_squares += num**2
        self.num_items += 1
        return self

    def remove(self, num: float) -> SumSquaresMeter:
        """Updates the statistic by removing a particular value.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        self.sum_squares -= num**2
        self.num_items -= 1
        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        return self.sum_squares


class VarianceMeter(RemovableStatistic):
    """The statistic that keeps track of the (population) variance of the values, using Welford's update.

    Welford's update tracks the mean and the sum of squared deviations from it, instead of the sum and the sum of squares, so the variance never comes out negative and keeps its precision on large values.

    Attributes:
        count (int): The number of values.
        mean (float): The mean of the values.
        m2 (float): The sum of squared deviations of the values from their mean.
    """

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, num: float) -> VarianceMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        self.count += 1
        delta = num - self.mean
        self.mean = self.mean + delta / self.count
        # The new mean lies between the old mean and num, so the increment is never negative.
        self.m2 = self.m2 + delta * (num - self.mean)

        return self

    def remove(self, num: float) -> VarianceMeter:
        """Updates the statistic by removing particular value.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        self.count -= 1
        if self.count == 0:
            self.mean = 0.0
            self.m2 = 0.0
            return self

        delta = num - self.mean
        self.mean = self.mean - delta / self.count
        # Undoing an update can round below zero when the remaining values are (nearly) equal.
        self.m2 = max(self.m2 - delta * (num - self.mean), 0.0)

        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        return self.m2 / self.count
