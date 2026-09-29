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


def _apply_threshold(y_pred: list[float], threshold: float | None) -> list[float] | list[int]:
    """Turns anomaly scores into 0/1 predictions when a threshold is given.

    Args:
        y_pred (list[float]): Predicted classes or scores.
        threshold (float | None): The score at or above which an instance is predicted anomalous. If None, y_pred is returned unchanged.

    Returns:
        list[float] | list[int]: The predicted classes.
    """
    if threshold is None:
        return y_pred

    return [1 if score >= threshold else 0 for score in y_pred]


class PrecisionMetric(BaseSKLearnMetric):
    """Precision wrapper class for sklearn.

    Precision is defined on predicted classes. With the default ``threshold=None``, ``y_pred`` must be 0 or 1 and is used as given. To pass anomaly scores, set ``threshold``. Scores at or above it are predicted anomalous (1) and the rest normal (0).

    Args:
        threshold (float | None): The score at or above which an instance is predicted anomalous. None expects 0/1 predictions. (Default=None).
    """

    def __init__(self, threshold: float | None = None) -> None:
        super().__init__()
        self.threshold = threshold

    def _evaluate(self, y_true: list[int], y_pred: list[float]) -> float:
        if not y_true:
            return 0.0
        return precision_score(y_true, _apply_threshold(y_pred, self.threshold))


class RecallMetric(BaseSKLearnMetric):
    """Recall wrapper class for sklearn.

    Recall is defined on predicted classes. With the default ``threshold=None``, ``y_pred`` must be 0 or 1 and is used as given. To pass anomaly scores, set ``threshold``. Scores at or above it are predicted anomalous (1) and the rest normal (0).

    Args:
        threshold (float | None): The score at or above which an instance is predicted anomalous. None expects 0/1 predictions. (Default=None).
    """

    def __init__(self, threshold: float | None = None) -> None:
        super().__init__()
        self.threshold = threshold

    def _evaluate(self, y_true: list[int], y_pred: list[float]) -> float:
        if not y_true:
            return 0.0
        return recall_score(y_true, _apply_threshold(y_pred, self.threshold))


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
        if not y_true:
            raise ValueError("No samples recorded. PR AUC score is not defined in that case.")
        return average_precision_score(y_true, y_pred)
