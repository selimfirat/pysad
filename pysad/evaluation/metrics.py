from __future__ import annotations

from abc import ABCMeta, abstractmethod

from sklearn.metrics import average_precision_score, roc_auc_score

from pysad.core.base_metric import BaseMetric


class BaseSKLearnMetric(BaseMetric, metaclass=ABCMeta):
    """Abstract base class to wrap the sklearn metrics."""

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


class _BaseConfusionMetric(BaseMetric):
    """Accumulate binary confusion counts without retaining samples."""

    def __init__(self, threshold: float | None = None) -> None:
        super().__init__()
        self.threshold = threshold
        self.true_positives_ = 0
        self.false_positives_ = 0
        self.false_negatives_ = 0
        self.invalid_labels_ = False

    def update(self, y_true: int, y_pred: float) -> None:
        """Record a binary label and prediction, applying the threshold on arrival.

        Args:
            y_true (int): Ground truth class, either 0 or 1.
            y_pred (float): Predicted class, or anomaly score when a threshold is set.
        """
        if y_true not in (0, 1) or (self.threshold is None and y_pred not in (0, 1)):
            self.invalid_labels_ = True
            return

        predicted = y_pred == 1 if self.threshold is None else y_pred >= self.threshold
        if y_true == 1:
            if predicted:
                self.true_positives_ += 1
            else:
                self.false_negatives_ += 1
        elif predicted:
            self.false_positives_ += 1

    def _ratio(self, denominator: int) -> float:
        if self.invalid_labels_:
            raise ValueError("Precision and recall require binary labels and predictions (0 or 1).")
        return self.true_positives_ / denominator if denominator else 0.0


class PrecisionMetric(_BaseConfusionMetric):
    """Streaming precision from binary confusion counts.

    Precision is defined on predicted classes. With the default ``threshold=None``, ``y_pred`` must be 0 or 1 and is used as given. To pass anomaly scores, set ``threshold``. Scores at or above it are predicted anomalous (1) and the rest normal (0).

    Only confusion counts are retained, so updates and queries take constant time. The threshold is applied when each sample is recorded. Calling ``get()`` before any samples have been recorded, or when the denominator is zero, returns ``0.0``.

    Args:
        threshold (float | None): The score at or above which an instance is predicted anomalous. None expects 0/1 predictions. (Default=None).
    """

    def get(self) -> float:
        """Return true positives divided by all predicted positives."""
        return self._ratio(self.true_positives_ + self.false_positives_)


class RecallMetric(_BaseConfusionMetric):
    """Streaming recall from binary confusion counts.

    Recall is defined on predicted classes. With the default ``threshold=None``, ``y_pred`` must be 0 or 1 and is used as given. To pass anomaly scores, set ``threshold``. Scores at or above it are predicted anomalous (1) and the rest normal (0).

    Only confusion counts are retained, so updates and queries take constant time. The threshold is applied when each sample is recorded. Calling ``get()`` before any samples have been recorded, or when the denominator is zero, returns ``0.0``.

    Args:
        threshold (float | None): The score at or above which an instance is predicted anomalous. None expects 0/1 predictions. (Default=None).
    """

    def get(self) -> float:
        """Return true positives divided by all actual positives."""
        return self._ratio(self.true_positives_ + self.false_negatives_)


class AUROCMetric(BaseSKLearnMetric):
    """Area under ROC curve wrapper class for sklearn.

    Calling ``get()`` before any samples have been recorded, or when fewer than two distinct classes are present in the recorded labels, raises a :class:`ValueError`.
    """

    def _evaluate(self, y_true: list[int], y_pred: list[float]) -> float:
        # Check if only one class is present
        if len(set(y_true)) <= 1:
            raise ValueError(
                "Only one class present in y_true. ROC AUC score is not defined in that case."
            )
        return roc_auc_score(y_true, y_pred)


class AUPRMetric(BaseSKLearnMetric):
    """Area under PR curve wrapper class for sklearn.

    Calling ``get()`` before any samples have been recorded raises a :class:`ValueError`.
    """

    def _evaluate(self, y_true: list[int], y_pred: list[float]) -> float:
        if not y_true:
            raise ValueError("No samples recorded. PR AUC score is not defined in that case.")
        return average_precision_score(y_true, y_pred)
