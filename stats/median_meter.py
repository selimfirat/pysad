from heapq import heappush

from stats.univariate_statistic import UnivariateStatistic


class MedianMeter(UnivariateStatistic):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.lst = []
        self.num_items = 0

    def update(self, num):
        heappush(self.lst, num)
        self.num_items += 1

        return self

    def remove(self, num):

        self.lst.remove(num)
        self.num_items -= 1


        return self

    def get(self):

        if self.num_items % 2 == 0:
            return (self.lst[self.num_items // 2] + self.lst[self.num_items//2 - 1]) / 2
        else:
            return self.lst[self.num_items//2]
