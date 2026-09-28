from __future__ import annotations

from pysad.core.base_statistic import BaseStatistic, UnivariateStatistic


class RunningStatistic(BaseStatistic):
    """The running statistic that wraps any other statistics to track statistics with a fixed window size.

    Args:
        statistic_cls (class): The class to be instantiated and to be windowed.
        window_size (int): The window size. Must be at least 1.
        **kwargs (Keyword arguments): The keyword arguments that is input to the statistic_cls.

    Raises:
        ValueError: If window_size is less than 1.
    """

    def __init__(
        self,
        statistic_cls: type[UnivariateStatistic],
        window_size: int,
        **kwargs
    ):
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")

        self.statistic_cls = statistic_cls
        self.statistic = self.statistic_cls(**kwargs)

        self.window_size = window_size
        self.window: list[float] = []

    def update(self, num: float) -> RunningStatistic:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        self.window.append(num)

        self.statistic.update(num)

        if len(self.window) > self.window_size:
            self.statistic.remove(self.window[0])
            self.window = self.window[1:]

        return self

    def get(self) -> float:
        """ Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        return self.statistic.get()
