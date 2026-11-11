from __future__ import annotations

from collections.abc import Iterable

from pysad.core.base_metric import BaseMetric
from pysad.evaluation.metrics import _confusion_counts_batch


class BalancedAccuracyMetric(BaseMetric):
    """Compute mean recall across the true classes in a binary stream.

    With both true classes present, the result is the mean of sensitivity
    and specificity. With one true class present, the result is its recall.
    These nonempty results match scikit-learn's ``balanced_accuracy_score``.
    Before any updates, ``get()`` returns ``0.0``.

    The metric stores four confusion counts instead of labels or scores.
    Memory use and the cost of each ``update()`` and ``get()`` stay constant
    as the stream grows.

    Args:
        threshold (float | None): Scores at or above this value predict an
            anomaly (1). Lower scores predict a normal instance (0).
            None requires 0/1 predictions (Default=None).
    """

    def __init__(self, threshold: float | None = None) -> None:
        super().__init__()
        self.threshold = threshold
        self.tp_ = 0
        self.tn_ = 0
        self.fp_ = 0
        self.fn_ = 0

    def update(self, y_true: int, y_pred: float) -> None:
        """Record one true label and prediction.

        Args:
            y_true (int): True class, 0 for normal or 1 for anomalous.
            y_pred (float): Predicted class, or score when a threshold exists.

        Raises:
            ValueError: The true label or predicted class is not 0 or 1.
        """
        if y_true not in (0, 1):
            raise ValueError("y_true must be 0 or 1.")
        if self.threshold is not None:
            y_pred = int(y_pred >= self.threshold)
        elif y_pred not in (0, 1):
            raise ValueError("y_pred must be 0 or 1 when threshold is None.")

        if y_true == 1:
            if y_pred == 1:
                self.tp_ += 1
            else:
                self.fn_ += 1
        elif y_pred == 1:
            self.fp_ += 1
        else:
            self.tn_ += 1

    def update_batch(
        self, y_true: Iterable[int], y_pred: Iterable[float]
    ) -> BalancedAccuracyMetric:
        """Record batches of true labels and predictions or scores."""
        counts = _confusion_counts_batch(y_true, y_pred, self.threshold)
        if counts is None:
            super().update_batch(y_true, y_pred)
            return self
        tp, fp, fn, tn = counts
        self.tp_ += tp
        self.fp_ += fp
        self.fn_ += fn
        self.tn_ += tn
        return self

    def get(self) -> float:
        """Return balanced accuracy without changing the counts.

        Returns:
            float: Mean recall across true classes, or 0.0 for an empty stream.
        """
        positives = self.tp_ + self.fn_
        negatives = self.tn_ + self.fp_
        if not positives:
            return self.tn_ / negatives if negatives else 0.0
        if not negatives:
            return self.tp_ / positives
        return (self.tp_ / positives + self.tn_ / negatives) / 2.0
