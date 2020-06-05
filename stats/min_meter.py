import math
from heapq import heappush

from stats.univariate_statistic import UnivariateStatistic


class MinMeter(UnivariateStatistic):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.min = -math.inf

        self.lst = []

    def update(self, num):
        if num < self.min:
            self.min = num

        heappush(self.lst, num)

        return self

    def remove(self, num):

        self.lst.remove(num)

        if len(self.lst) > 0:
            self.min = self.lst[0]
        else:
            self.min = math.inf

        return self
