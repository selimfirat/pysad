from evaluation.base_metric import BaseMetric
from stats.mean_meter import MeanMeter


class WindowedMetric(BaseMetric):

    def __init__(self, metric_cls, window_size, ignore_nonempty_last=True, **kwargs):
        super().__init__(**kwargs)

        self.ignore_nonempty_last = ignore_nonempty_last
        self.window_size = window_size
        self.metric_cls = metric_cls

        self.metric = self.init_metric()

        self.score_meter = MeanMeter()

        self.step = 0

        self.num_windows = 0

    def init_metric(self):

        return self.metric_cls(**self.kwargs)

    def update(self, y_true, y_pred):

        self.step += 1
        self.metric.update(y_true, y_pred)

        if self.step % self.window_size == 0:
            self.num_windows += 1
            score = self.metric.get()
            self.score_meter.update(score)
            self.metric = self.init_metric()

        return self

    def get(self):

        if self.ignore_nonempty_last:
            return self.score_meter.get()
        else:
            return (self.metric.get() + self.score_meter.get()*self.num_windows) / (self.num_windows + 1)
