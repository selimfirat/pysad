from abc import abstractmethod, ABCMeta
from pysad.evaluation.base_metric import BaseMetric


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
