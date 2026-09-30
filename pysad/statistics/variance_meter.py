from __future__ import annotations

from pysad.core.base_statistic import UnivariateStatistic


class VarianceMeter(UnivariateStatistic):
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
