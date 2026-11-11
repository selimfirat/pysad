from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from pysad.core.base_metric import BaseMetric
from pysad.evaluation.metrics import _ConfusionMatrixMetric


class RollingMetric(BaseMetric):
    """Evaluate a metric on the most recent label-score pairs.

    Each update adds one pair and drops the oldest pair when the window is full.
    An empty window returns None. A partial window uses all available pairs.
    Nonempty windows return the selected metric's result or raise its errors.
    For example, AUROCMetric raises ValueError when a window contains only one class.

    Storage uses O(window_size) pairs in preallocated NumPy ring arrays, and updates take O(1) time.
    Confusion-matrix metrics (precision, recall, F1, balanced accuracy, MCC) maintain sliding
    confusion counts so ``get()`` runs in O(1) time; AUROCMetric and AUPRMetric sort scores in
    O(window_size * log(window_size)) time.

    Unlike this sliding window, :class:`~pysad.evaluation.WindowedMetric`
    averages metric values from consecutive windows that do not overlap.

    Args:
        metric_cls (type[BaseMetric]): The metric class to evaluate.
        window_size (int): The maximum number of recent pairs to retain.
        **kwargs (Any): Constructor arguments for each fresh metric.
    """

    def __init__(
        self,
        metric_cls: type[BaseMetric],
        window_size: int,
        **kwargs: Any,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")

        super().__init__()
        self.metric_cls = metric_cls
        self.window_size = window_size
        self.metric_kwargs = kwargs

        self._yt_buf = np.empty(window_size, dtype=np.float64)
        self._yp_buf = np.empty(window_size, dtype=np.float64)
        self._count = 0
        self._head = 0

        self._is_confusion = issubclass(metric_cls, _ConfusionMatrixMetric)
        self._threshold: float | None = kwargs.get("threshold")
        self._bucket_buf = np.empty(window_size, dtype=np.int8) if self._is_confusion else None
        self._counts = [0, 0, 0, 0]  # [tp, fp, fn, tn]
        self._invalid_count = 0

    def _classify_pair(self, y_true: int, y_pred: float) -> int:
        if y_true not in (0, 1):
            return -1
        if self._threshold is None:
            if y_pred not in (0, 1):
                return -1
            pred = y_pred == 1
        else:
            pred = y_pred >= self._threshold
        if y_true == 1:
            return 0 if pred else 2
        return 1 if pred else 3

    def update(self, y_true: int, y_pred: float) -> RollingMetric:
        """Add a label-score pair to the recent window.

        Args:
            y_true (int): The true label, 1 for an anomaly and 0 otherwise.
            y_pred (float): The predicted score or label for this instance.

        Returns:
            RollingMetric: This metric.
        """
        slot = self._head
        if self._is_confusion:
            assert self._bucket_buf is not None
            if self._count == self.window_size:
                old_b = int(self._bucket_buf[slot])
                if old_b < 0:
                    self._invalid_count -= 1
                else:
                    self._counts[old_b] -= 1
            new_b = self._classify_pair(y_true, y_pred)
            self._bucket_buf[slot] = new_b
            if new_b < 0:
                self._invalid_count += 1
            else:
                self._counts[new_b] += 1

        self._yt_buf[slot] = y_true
        self._yp_buf[slot] = y_pred
        self._head = (slot + 1) % self.window_size
        if self._count < self.window_size:
            self._count += 1
        return self

    def update_batch(self, y_true: Iterable[int], y_pred: Iterable[float]) -> RollingMetric:
        """Add batches of label-score pairs to the recent window."""
        yt_arr = np.asarray(y_true if isinstance(y_true, np.ndarray) else list(y_true))
        yp_arr = np.asarray(y_pred if isinstance(y_pred, np.ndarray) else list(y_pred))
        if len(yt_arr) != len(yp_arr):
            raise ValueError("y_true and y_pred must have the same length.")
        n = len(yt_arr)
        if n == 0:
            return self
        if n >= self.window_size:
            yt_tail = np.asarray(yt_arr[-self.window_size :], dtype=np.float64)
            yp_tail = np.asarray(yp_arr[-self.window_size :], dtype=np.float64)
            self._yt_buf[:] = yt_tail
            self._yp_buf[:] = yp_tail
            self._count = self.window_size
            self._head = 0
            if self._is_confusion:
                assert self._bucket_buf is not None
                self._counts = [0, 0, 0, 0]
                self._invalid_count = 0
                for idx in range(self.window_size):
                    yt_val = float(yt_tail[idx])
                    b = self._classify_pair(
                        int(yt_val) if yt_val in (0.0, 1.0) else -1, float(yp_tail[idx])
                    )
                    self._bucket_buf[idx] = b
                    if b < 0:
                        self._invalid_count += 1
                    else:
                        self._counts[b] += 1
            return self

        for yt, yp in zip(yt_arr.tolist(), yp_arr.tolist(), strict=True):
            self.update(yt, yp)
        return self

    def _ordered_window(self) -> tuple[np.ndarray, np.ndarray]:
        if self._count < self.window_size or self._head == 0:
            return self._yt_buf[: self._count], self._yp_buf[: self._count]
        h = self._head
        return (
            np.concatenate((self._yt_buf[h:], self._yt_buf[:h])),
            np.concatenate((self._yp_buf[h:], self._yp_buf[:h])),
        )

    def get(self) -> float | None:
        """Recompute the metric without changing the window.

        Returns:
            float | None: The metric result, or None for an empty window.
        """
        if self._count == 0:
            return None

        metric = self.metric_cls(**self.metric_kwargs)
        if self._is_confusion and self._invalid_count == 0:
            assert isinstance(metric, _ConfusionMatrixMetric)
            metric.tp_, metric.fp_, metric.fn_, metric.tn_ = self._counts
            return metric.get()

        yt_win, yp_win = self._ordered_window()
        metric.update_batch(yt_win, yp_win)
        return metric.get()
