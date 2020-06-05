

from stats.univariate_statistic import UnivariateStatistic


class SumSquaresMeter(UnivariateStatistic):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sum_squares = 0.0
        self.num_items = 0

    def update(self, num):

        self.sum_squares += num**2
        self.num_items += 1

        return self

    def remove(self, num):

        self.sum_squares -= num**2
        self.num_items -= 1

        return self

    def get(self):

        return self.sum_squares
