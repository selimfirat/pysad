from pysad.core.base_statistic import UnivariateStatistic


class AbsStatistic(UnivariateStatistic):
    """The absolute value of the statistic that is tracked.

    Args:
        statistic_cls (class): The class of the statistic to be instiantiated.
        **kwargs (Keyword arguments): The keyword arguments that is input to the statistic_cls.
    """

    def __init__(self, statistic_cls, **kwargs):
        self.statistic_cls = statistic_cls

        self.statistic = self.statistic_cls(**kwargs)

    def update(self, num):
        """Updates the statistic with the value for a timestep.

        Args:
            num (float):  The incoming value, for which the statistic is used.

        Returns:
            object: self.

        """
        self.statistic.update(num)

        return self

    def remove(self, num):
        """Updates the statistic by removing particular value. This method

        Args:
            num (float):  The value to be removed.

        Returns:
            object: self.

        """
        self.statistic.remove(num)

        return self

    def get(self):
        """Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        return abs(self.statistic.get())
