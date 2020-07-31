from pysad.stats.base_statistic import UnivariateStatistic


class AbsStatistic(UnivariateStatistic):

    def __init__(self, statistic_cls, **kwargs):
        """The absolute value of the statistic that is tracked.

        Args:
            statistic_cls: The class of the statistic to be instiantiated.
            **kwargs: The keyword arguments that is input to the statistic_cls.
        """
        self.statistic_cls = statistic_cls

        self.statistic = self.statistic_cls(**kwargs)

    def update(self, num):
        """Updates the statistic with the value for a timestep.

        Args:
            num: The incoming value, for which the statistic is used.

        Returns:
            self: object
                Returns the fitted statistic.

        """
        self.statistic.update(num)

        return self

    def remove(self, num):
        """Updates the statistic by removing particular value. This method

        Args:
            num: The value to be removed.

        Returns:
            self: object
                Returns the fitted statistic.

        """
        self.statistic.remove(num)

        return self

    def get(self):
        """ Method to obtain the tracked statistic.

        Returns:
            statistic: float
                The statistic.
        """
        return abs(self.statistic.get())
