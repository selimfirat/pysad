import abc
from abc import abstractmethod


class BaseStatistic(abc.ABC):
    """Abstact base class for the statistics.
    """
    pass


class UnivariateStatistic(BaseStatistic):
    """Abstract base class for univariate statistics.
    """

    @abstractmethod
    def update(self, num):
        """Updates the statistic with the value for a timestep.

        Args:
            num: The incoming value, for which the statistic is used.

        Returns:
            self: object
                Returns the fitted statistic.

        """
        pass

    @abstractmethod
    def get(self):
        """ Method to obtain the tracked statistic.

        Returns:
            statistic: float
                The statistic.
        """
        pass

    @abstractmethod
    def remove(self, num):
        """Updates the statistic by removing particular value. This method

        Args:
            num: The value to be removed.

        Returns:
            self: object
                Returns the fitted statistic.

        """
        pass
