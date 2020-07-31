import math
from heapq import heappush

from pysad.stats.univariate_statistic import UnivariateStatistic


class MaxMeter(UnivariateStatistic):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max = -math.inf

        self.lst = []

    def update(self, num):
        if num > self.max:
            self.max = num

        heappush(self.lst, num)

        return self

    def remove(self, num):

        self.lst.remove(num)

        if len(self.lst) > 0:
            self.max = self.lst[-1]
        else:
            self.max = -math.inf

        return self
