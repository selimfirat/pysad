from stats.base_statistic import BaseStatistic


class RunningStatistic(BaseStatistic):

    def __init__(self, statistic_cls, window_size, **kwargs):
        super().__init__(**kwargs)
        self.statistic_cls = statistic_cls
        self.statistic = self.statistic_cls()

        self.window_size = window_size
        self.window = []

    def update(self, num):

        self.window.append(num)

        self.statistic.update(num)

        if len(self.window) >= self.window_size:
            self.statistic.remove(num)
            self.window = self.window[1:]

        return self

    def get(self):

        return self.statistic.get()
