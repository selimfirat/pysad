from __future__ import annotations

from pysad.core.base_metric import BaseMetric


class F1Metric(BaseMetric):
    """Binary F1 score over a stream, using constant memory.

    F1 is ``2 * TP / (2 * TP + FP + FN)`` for the anomaly class (1).
    Only true positive, false positive and false negative counts are retained;
    both ``update()`` and ``get()`` take constant time.

    As with :class:`PrecisionMetric`, ``threshold=None`` expects 0/1 predictions.
    With a threshold, scores at or above it predict an anomaly (1), and all other
    scores predict a normal instance (0). The threshold applies when updating.

    ``get()`` returns ``0.0`` before any updates or when the denominator is zero
    (no true or predicted anomalies), matching scikit-learn with
    ``zero_division=0``. Reading the score does not change the counts.

    Args:
        threshold (float | None): The score at or above which an instance is
            predicted anomalous. None expects 0/1 predictions. (Default=None).
    """

    def __init__(self, threshold: float | None = None) -> None:
        super().__init__()
        self.threshold = threshold
        self.tp_ = 0
        self.fp_ = 0
        self.fn_ = 0

    def update(self, y_true: int, y_pred: float) -> None:
        """Record a label and prediction, or a score when a threshold is set.

        Args:
            y_true (int): Ground truth class, 1 for an anomaly or 0 for normal.
            y_pred (float): Binary prediction, or anomaly score with a threshold.

        Raises:
            ValueError: If the label is not binary, or the prediction is not
                binary when no threshold is set.
        """
        if y_true not in (0, 1):
            raise ValueError("y_true must be 0 or 1.")
        if self.threshold is None:
            if y_pred not in (0, 1):
                raise ValueError("y_pred must be 0 or 1 when threshold is None.")
            predicted = y_pred
        else:
            predicted = int(y_pred >= self.threshold)

        if predicted == 1:
            if y_true == 1:
                self.tp_ += 1
            else:
                self.fp_ += 1
        elif y_true == 1:
            self.fn_ += 1

    def get(self) -> float:
        """Return the current F1 score, or 0.0 if its denominator is zero.

        Returns:
            float: The F1 score for all recorded updates.
        """
        denominator = 2 * self.tp_ + self.fp_ + self.fn_
        if denominator == 0:
            return 0.0
        return 2 * self.tp_ / denominator
