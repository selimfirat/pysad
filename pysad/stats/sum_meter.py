from pysad.stats.univariate_statistic import UnivariateStatistic


class SumMeter(UnivariateStatistic):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sum = 0.0
        self.num_items = 0

    def update(self, num):

        self.sum += num
        self.num_items += 1

        return self

    def remove(self, num):

        self.sum -= num
        self.num_items -= 1

        return self

    def get(self):

        return self.sum
