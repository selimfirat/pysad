from pysad.core.base_statistic import BaseStatistic


class RunningStatistic(BaseStatistic):
    """The running statistic that wraps any other statistics to track statistics with a fixed window size.

    Args:
        statistic_cls (class): The class to be instantiated and to be windowed.
        window_size (int): The window size.
        **kwargs (Keyword arguments): The keyword arguments that is input to the statistic_cls.
    """

    def __init__(self, statistic_cls, window_size, **kwargs):
        super().__init__(**kwargs)
        self.statistic_cls = statistic_cls
        self.statistic = self.statistic_cls()

        self.window_size = window_size
        self.window = []

    def update(self, num):
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

    def get(self):
        """ Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        return self.statistic.get()
