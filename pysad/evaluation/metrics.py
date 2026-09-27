from __future__ import annotations

from abc import ABCMeta, abstractmethod
from sklearn.metrics import recall_score, precision_score, roc_auc_score, average_precision_score
from pysad.core.base_metric import BaseMetric


class BaseSKLearnMetric(BaseMetric, metaclass=ABCMeta):
    """Abstract base class to wrap the sklearn metrics.
    """

    def __init__(self) -> None:
        self.y_true: list[int] = []
        self.y_pred: list[float] = []

    def update(self, y_true: int, y_pred: float) -> None:
        """Updates the metric with given true and predicted value for a timestep.

        Args:
            y_true (int): Ground truth class. Either 1 or 0.
            y_pred (float): Predicted class or anomaly score. Higher values correspond to more anomalousness and lower values correspond to more normalness.
        """
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def get(self) -> float:
        """Gets the current value of the score.

        Returns:
            float: The current score.
        """
        score = self._evaluate(self.y_true, self.y_pred)

        return score

    @abstractmethod
    def _evaluate(self, y_true: list[int], y_pred: list[float]) -> float:
        """Abstract method to be filled with the sklearn metric.

        Args:
            y_true (list[int]): Ground truth classes.
            y_pred (list[float]): Predicted classes or scores.
        """
        pass


class PrecisionMetric(BaseSKLearnMetric):
    """Precision wrapper class for sklearn.
    """

    def _evaluate(self, y_true: list[int], y_pred: list[float]) -> float:
        return precision_score(y_true, y_pred)


class RecallMetric(BaseSKLearnMetric):
    """Recall wrapper class for sklearn.
    """

    def _evaluate(self, y_true: list[int], y_pred: list[float]) -> float:
        return recall_score(y_true, y_pred)


class AUROCMetric(BaseSKLearnMetric):
    """Area under roc curve wrapper class for sklearn.
    """

    def _evaluate(self, y_true: list[int], y_pred: list[float]) -> float:
        # Check if only one class is present
        if len(set(y_true)) <= 1:
            raise ValueError("Only one class present in y_true. ROC AUC score is not defined in that case.")
        return roc_auc_score(y_true, y_pred)


class AUPRMetric(BaseSKLearnMetric):
    """Area under PR curve wrapper class for sklearn.
    """

    def _evaluate(self, y_true: list[int], y_pred: list[float]) -> float:
        return average_precision_score(y_true, y_pred)
