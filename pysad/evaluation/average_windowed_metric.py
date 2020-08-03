from pysad.evaluation.base_metric import BaseMetric
from pysad.stats.average_meter import AverageMeter


class WindowedMetric(BaseMetric):
    """A helper class to evlauate windowed metrics. The distributions of the streaming model scores often change due to model collapse (i.e. becoming closer to the always loss=0) or appearing nonstationarities. Thus, the metrics such as ROC or AUC scores may change drastically. To prevent their effect, this class creates windows of size `window_size`. After each `window_size`th object, a new instance of the `metric_cls` is being created. Lastly, the metrics from all windows are averaged :cite:`xstream,gokcesu2017online`.

    Args:
        metric_cls: class
            The metric class to be windowed.
        window_size: int
            The window size.
        ignore_nonempty_last: bool
            Whether to ignore the score of the nonempty last window. Note that the empty last window is always ignored.

    """

    def __init__(self, metric_cls, window_size, ignore_nonempty_last=True):
        super().__init__()

        self.ignore_nonempty_last = ignore_nonempty_last
        self.window_size = window_size
        self.metric_cls = metric_cls

        self.metric = self.init_metric()

        self.score_meter = AverageMeter()

        self.step = 0

        self.num_windows = 0

    def _init_metric(self):

        return self.metric_cls()

    def update(self, y_true, y_pred):
        """

        Args:
            y_true: float
                The ground truth score for the incoming instance.
            y_pred: float
                The predicted score for the incoming instance.

        Returns:
            self: object
                The updated metric.
        """
        self.step += 1
        self.metric.update(y_true, y_pred)

        if self.step % self.window_size == 0:
            self.num_windows += 1
            score = self.metric.get()
            self.score_meter.update(score)
            self.metric = self._init_metric()

        return self

    def get(self):
        """Obtains the averaged score.

        Returns:
            current_score: float
                The average score of the windows.
        """
        if self.ignore_nonempty_last:
            return self.score_meter.get()
        else:
            return (self.metric.get() + self.score_meter.get()*self.num_windows) / (self.num_windows + 1)
