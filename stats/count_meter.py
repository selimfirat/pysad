from stats.univariate_statistic import UnivariateStatistic


class CountMeter(UnivariateStatistic):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.count = 0

    def update(self, num=None):

        self.count += 1

        return self

    def remove(self, num=None):

        self.count -= 1

        return self

    def get(self):

        return self.count
