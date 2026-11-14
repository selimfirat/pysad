from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from math import sqrt

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from pysad.core.base_metric import BaseMetric


def _confusion_counts_batch(
    y_true: Iterable[int],
    y_pred: Iterable[float],
    threshold: float | None,
) -> tuple[int, int, int, int] | None:
    """Validates binary inputs and returns ``(tp, fp, fn, tn)``, or ``None`` if non-numeric."""
    yt_arr = y_true if isinstance(y_true, np.ndarray) else np.asarray(list(y_true))
    yp_arr = y_pred if isinstance(y_pred, np.ndarray) else np.asarray(list(y_pred))
    if yt_arr.ndim != 1 or yp_arr.ndim != 1:
        if len(yt_arr) != len(yp_arr):
            raise ValueError("y_true and y_pred must have the same length.")
        return None
    if len(yt_arr) != len(yp_arr):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(yt_arr) == 0:
        return (0, 0, 0, 0)
    if yt_arr.dtype.kind not in "biuf" or yp_arr.dtype.kind not in "biuf":
        return None

    yt_valid = (yt_arr == 0) | (yt_arr == 1)
    if threshold is None:
        yp_valid = (yp_arr == 0) | (yp_arr == 1)
        if not bool(yt_valid.all()):
            if bool(yp_valid.all()) or int(np.flatnonzero(~yt_valid)[0]) <= int(
                np.flatnonzero(~yp_valid)[0]
            ):
                raise ValueError("y_true must be 0 or 1.")
            raise ValueError("y_pred must be 0 or 1 when threshold is None.")
        if not bool(yp_valid.all()):
            raise ValueError("y_pred must be 0 or 1 when threshold is None.")
        pred_pos = yp_arr == 1
    else:
        if not bool(yt_valid.all()):
            raise ValueError("y_true must be 0 or 1.")
        pred_pos = yp_arr >= threshold

    actual_pos = yt_arr == 1
    n_pos = int(np.count_nonzero(actual_pos))
    n_pred = int(np.count_nonzero(pred_pos))
    tp = int(np.count_nonzero(actual_pos & pred_pos))
    fp = n_pred - tp
    fn = n_pos - tp
    tn = len(yt_arr) - n_pos - fp
    return (tp, fp, fn, tn)


class BaseSKLearnMetric(BaseMetric, metaclass=ABCMeta):
    """Abstract base class for ranking metrics that require full score history."""

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

    def update_batch(self, y_true: Iterable[int], y_pred: Iterable[float]) -> BaseSKLearnMetric:
        """Appends batches of true labels and predicted values in stream order."""
        yt_list = y_true.tolist() if isinstance(y_true, np.ndarray) else list(y_true)
        yp_list = y_pred.tolist() if isinstance(y_pred, np.ndarray) else list(y_pred)
        if len(yt_list) != len(yp_list):
            raise ValueError("y_true and y_pred must have the same length.")
        self.y_true.extend(yt_list)
        self.y_pred.extend(yp_list)
        return self

    def get(self) -> float:
        """Gets the current value of the score.

        Returns:
            float: The current score.
        """
        return self._evaluate(self.y_true, self.y_pred)

    @abstractmethod
    def _evaluate(self, y_true: list[int], y_pred: list[float]) -> float:
        """Abstract method to be filled with the sklearn metric.

        Args:
            y_true (list[int]): Ground truth classes.
            y_pred (list[float]): Predicted classes or scores.
        """
        pass


class _ConfusionMatrixMetric(BaseMetric, metaclass=ABCMeta):
    """Base class for binary classification metrics backed by O(1) confusion counts.

    Args:
        threshold (float or None): Scores at or above this value predict anomaly label 1. None expects 0/1 predictions (Default=None).
    """

    def __init__(self, threshold: float | None = None) -> None:
        super().__init__()
        self.threshold = threshold
        self.tp_ = 0
        self.tn_ = 0
        self.fp_ = 0
        self.fn_ = 0

    def update(self, y_true: int, y_pred: float) -> None:
        """Records one binary label and prediction (or thresholded score) in constant time.

        Args:
            y_true (int): True label, 1 for an anomaly or 0 for normal.
            y_pred (float): Binary prediction, or anomaly score when ``threshold`` is set.

        Raises:
            ValueError: If ``y_true`` is not 0/1, or ``y_pred`` is not 0/1 when ``threshold`` is None.
        """
        if y_true not in (0, 1):
            raise ValueError("y_true must be 0 or 1.")
        if self.threshold is None:
            if y_pred not in (0, 1):
                raise ValueError("y_pred must be 0 or 1 when threshold is None.")
            predicted = y_pred == 1
        else:
            predicted = y_pred >= self.threshold

        if y_true == 1:
            if predicted:
                self.tp_ += 1
            else:
                self.fn_ += 1
        elif predicted:
            self.fp_ += 1
        else:
            self.tn_ += 1

    def update_batch(
        self, y_true: Iterable[int], y_pred: Iterable[float]
    ) -> _ConfusionMatrixMetric:
        """Records batches of binary labels and predictions or scores in vectorized form."""
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


class PrecisionMetric(_ConfusionMatrixMetric):
    """Binary precision over a stream using constant memory and constant work per update.

    Precision is ``TP / (TP + FP)`` for the anomaly class (1). With the default ``threshold=None``,
    ``y_pred`` must be 0 or 1. To pass anomaly scores, set ``threshold`` so scores at or above it
    predict an anomaly (1). Calling ``get()`` before any updates or with no positive predictions
    returns ``0.0``.

    Args:
        threshold (float or None): The score at or above which an instance is predicted anomalous. None expects 0/1 predictions (Default=None).
    """

    def get(self) -> float:
        """Returns current precision without changing the counts."""
        predicted_pos = self.tp_ + self.fp_
        return self.tp_ / predicted_pos if predicted_pos else 0.0


class RecallMetric(_ConfusionMatrixMetric):
    """Binary recall over a stream using constant memory and constant work per update.

    Recall is ``TP / (TP + FN)`` for the anomaly class (1). With the default ``threshold=None``,
    ``y_pred`` must be 0 or 1. To pass anomaly scores, set ``threshold`` so scores at or above it
    predict an anomaly (1). Calling ``get()`` before any updates or with no true anomalies
    returns ``0.0``.

    Args:
        threshold (float or None): The score at or above which an instance is predicted anomalous. None expects 0/1 predictions (Default=None).
    """

    def get(self) -> float:
        """Returns current recall without changing the counts."""
        positives = self.tp_ + self.fn_
        return self.tp_ / positives if positives else 0.0


class F1Metric(_ConfusionMatrixMetric):
    """Binary F1 score over a stream using constant memory and constant work per update.

    F1 is ``2 * TP / (2 * TP + FP + FN)`` for the anomaly class (1). With ``threshold=None``,
    ``y_pred`` must be 0 or 1. With a threshold, scores at or above it predict an anomaly (1).
    ``get()`` returns ``0.0`` before any updates or when the denominator is zero, matching
    scikit-learn with ``zero_division=0``.

    Args:
        threshold (float or None): The score at or above which an instance is predicted anomalous. None expects 0/1 predictions (Default=None).
    """

    def get(self) -> float:
        """Returns the current F1 score, or 0.0 if its denominator is zero."""
        denominator = 2 * self.tp_ + self.fp_ + self.fn_
        return 2 * self.tp_ / denominator if denominator else 0.0


class BalancedAccuracyMetric(_ConfusionMatrixMetric):
    """Mean recall across the true classes in a binary stream using constant memory.

    With both true classes present, the result is the mean of sensitivity and specificity. With
    one true class present, the result is its recall, matching scikit-learn's
    ``balanced_accuracy_score``. Before any updates, ``get()`` returns ``0.0``.

    Args:
        threshold (float or None): Scores at or above this value predict an anomaly (1). None requires 0/1 predictions (Default=None).
    """

    def get(self) -> float:
        """Returns balanced accuracy without changing the counts."""
        positives = self.tp_ + self.fn_
        negatives = self.tn_ + self.fp_
        if not positives:
            return self.tn_ / negatives if negatives else 0.0
        if not negatives:
            return self.tp_ / positives
        return (self.tp_ / positives + self.tn_ / negatives) / 2.0


class MatthewsCorrelationMetric(_ConfusionMatrixMetric):
    """Matthews correlation coefficient for binary labels and predictions using constant memory.

    The result ranges from -1 for opposite predictions to 1 for perfect predictions. Zero means no
    correlation. An empty stream or a zero denominator returns ``0.0``.

    Args:
        threshold (float or None): Scores at or above this value predict anomaly label 1. None requires predictions of 0 or 1 (Default=None).
    """

    def get(self) -> float:
        """Returns the correlation without changing the counts."""
        denominator = (
            (self.tp_ + self.fp_)
            * (self.tp_ + self.fn_)
            * (self.tn_ + self.fp_)
            * (self.tn_ + self.fn_)
        )
        if denominator == 0:
            return 0.0
        return (self.tp_ * self.tn_ - self.fp_ * self.fn_) / sqrt(denominator)


class AUROCMetric(BaseSKLearnMetric):
    """Area under ROC curve wrapper class for sklearn.

    Calling ``get()`` before any samples have been recorded, or when fewer than two distinct classes are present in the recorded labels, raises a :class:`ValueError`.
    """

    def _evaluate(self, y_true: list[int], y_pred: list[float]) -> float:
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


class LogLossMetric(BaseMetric):
    r"""Prequential logarithmic loss metric for probability evaluation :cite:`cesabianchi2005learning`.

    Computes the running logarithmic loss (binary cross-entropy) of predicted anomaly
    probabilities with respect to observed binary ground truth classes. For each instance
    :math:`(y_t, p_t)` where :math:`y_t \in \{0, 1\}` and :math:`p_t \in [0, 1]`,
    the loss is:

    .. math::

        \ell(y_t, p_t) = -\left[ y_t \ln(p_t) + (1 - y_t) \ln(1 - p_t) \right]

    Lower loss indicates better calibration and prediction quality, with 0.0
    representing perfect prediction.

    Probabilities are clipped to :math:`[\epsilon, 1 - \epsilon]` to ensure numerical
    stability and finite loss when :math:`p_t \in \{0, 1\}`. Updates execute in
    :math:`\mathcal{O}(1)` time and memory.

    Args:
        eps (float): Epsilon clipping factor to bound probabilities away from 0 and 1.
            Defaults to 1e-15.
    """

    def __init__(self, eps: float = 1e-15) -> None:
        if eps <= 0.0 or eps >= 0.5:
            raise ValueError("eps must be strictly between 0 and 0.5.")
        super().__init__()
        self.eps = float(eps)
        self.total_loss: float = 0.0
        self.count: int = 0

    def update(self, y_true: int, y_pred: float) -> LogLossMetric:
        """Updates the metric with given true class and predicted anomaly probability.

        Args:
            y_true (int): Ground truth class. Either 1 or 0.
            y_pred (float): Predicted anomaly probability in [0, 1].

        Raises:
            ValueError: If `y_true` is not 0 or 1, or `y_pred` is outside [0, 1] or NaN.

        Returns:
            LogLossMetric: The metric itself.
        """
        if y_true not in (0, 1):
            raise ValueError("y_true must be 0 or 1.")
        yp = float(y_pred)
        if not (0.0 <= yp <= 1.0) or np.isnan(yp):
            raise ValueError("y_pred must be a probability between 0 and 1.")

        p = min(max(yp, self.eps), 1.0 - self.eps)
        loss = -np.log(p) if y_true == 1 else -np.log(1.0 - p)

        self.total_loss += float(loss)
        self.count += 1
        self.score = self.total_loss / self.count
        return self

    def get(self) -> float | None:
        """Gets the running average logarithmic loss.

        Returns:
            float | None: The current running log loss, or None if no instances have been evaluated.
        """
        return self.score
