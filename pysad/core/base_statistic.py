from __future__ import annotations

from abc import abstractmethod, ABC


class BaseStatistic(ABC):
    """Abstact base class for the statistics.
    """
    pass


class UnivariateStatistic(BaseStatistic):
    """Abstract base class for univariate statistics.
    """

    @abstractmethod
    def update(self, num: float) -> "UnivariateStatistic":
        """Updates the statistic with the value for a timestep.

        Args:
            num (float):  The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        pass

    @abstractmethod
    def get(self) -> float:
        """ Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        pass

    @abstractmethod
    def remove(self, num: float) -> "UnivariateStatistic":
        """Updates the statistic by removing a particular value.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        pass
