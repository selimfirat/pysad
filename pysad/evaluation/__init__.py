from pysad.stats.average_meter import AverageMeter
from abc import ABCMeta, abstractmethod
from sklearn.metrics import recall_score, precision_score, roc_auc_score, average_precision_score
from pysad.core.base_metric import BaseMetric


class WindowedMetric(BaseMetric):
    """A helper class to evaluate windowed metrics. The distributions of the streaming model scores often change due to model collapse (i.e. becoming closer to the always loss=0) or appearing nonstationarities. Thus, the metrics such as ROC or AUC scores may change drastically. To prevent their effect, this class creates windows of size `window_size`. After each `window_size`th object, a new instance of the `metric_cls` is being created. Lastly, the metrics from all windows are averaged :cite:`xstream,gokcesu2017online`.

    Args:
        metric_cls: class
            The metric class to be windowed.
        window_size: int
            The window size.
        ignore_nonempty_last: bool
            Whether to ignore the score of the nonempty last window. Note that the empty last window is always ignored.

    """

    def __init__(self, metric_cls, window_size, ignore_nonempty_last=True):
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


class BaseSKLearnMetric(BaseMetric, metaclass=ABCMeta):
    """Abstract base class to wrap the sklearn metrics.
    """

    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def update(self, y_true, y_pred):
        """Updates the metric with given true and predicted value for a timestep.

        Args:
            y_true: int
                Ground truth class. Either 1 or 0.
            y_pred: int
                Predicted class. Either 1 or 0.
        """
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def get(self):
        """Gets the current value of the score.

        Returns:
            score: float
                The current score.
        """
        score = self._evaluate(self.y_true, self.y_pred)

        return score

    @abstractmethod
    def _evaluate(self, y_true, y_pred):
        """Abstract method to be filled with the sklearn metric.

        Args:
            y_true: list[int]
                Ground truth classes.
            y_pred: list[int]
                Predicted classes.
        """
        pass


class PrecisionMetric(BaseSKLearnMetric):
    """Precision wrapper class for sklearn.
    """

    def _evaluate(self, y_true, y_pred):
        return precision_score(y_true, y_pred)


class RecallMetric(BaseSKLearnMetric):
    """Recall wrapper class for sklearn.
    """

    def _evaluate(self, y_true, y_pred):
        return recall_score(y_true, y_pred)


class AUROCMetric(BaseSKLearnMetric):
    """Area under roc curve wrapper class for sklearn.
    """

    def _evaluate(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)


class AUPRMetric(BaseSKLearnMetric):
    """Area under PR curve wrapper class for sklearn.
    """

    def _evaluate(self, y_true, y_pred):
        return average_precision_score(y_true, y_pred)
