from abc import abstractmethod

from stats.base_statistic import BaseStatistic


class UnivariateStatistic(BaseStatistic):

    @abstractmethod
    def update(self, num):
        pass

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def remove(self, num):
        pass