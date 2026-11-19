from __future__ import annotations

from abc import ABC, abstractmethod


class BaseStatistic(ABC):
    """Abstract base class for streaming univariate statistics."""

    _removable: bool = False

    @abstractmethod
    def update(self, num: float) -> BaseStatistic:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            BaseStatistic: self.
        """
        pass

    @abstractmethod
    def get(self) -> float:
        """Obtains the current value of the tracked statistic.

        Returns:
            float: The statistic.
        """
        pass


class RemovableStatistic(BaseStatistic):
    """Abstract base class for streaming statistics that support removing past observations."""

    _removable: bool = True

    @abstractmethod
    def remove(self, num: float) -> RemovableStatistic:
        """Updates the statistic by removing a particular value.

        Args:
            num (float): The value to be removed.

        Returns:
            RemovableStatistic: self.
        """
        pass
