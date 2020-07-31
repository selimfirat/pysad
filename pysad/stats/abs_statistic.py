from pysad.stats.univariate_statistic import UnivariateStatistic


class AbsStatistic(UnivariateStatistic):
    def __init__(self, statistic_cls, **kwargs):
        super().__init__(**kwargs)
        self.statistic_cls = statistic_cls

        self.statistic = self.statistic_cls(**kwargs)

    def update(self, num):
        self.statistic.update(num)

        return self

    def remove(self, num):
        self.statistic.remove(num)

        return self

    def get(self):

        return abs(self.statistic.get())
