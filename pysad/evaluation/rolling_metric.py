from __future__ import annotations

from collections import deque
from typing import Any

from pysad.core.base_metric import BaseMetric


class RollingMetric(BaseMetric):
    """Evaluate a metric on the most recent label-score pairs.

    Each update adds one pair and drops the oldest pair when the window is full.
    Each call to ``get()`` creates a fresh metric and replays the current window.
    An empty window returns None. A partial window uses all available pairs.
    Nonempty windows return the selected metric's result or raise its errors.
    For example, AUROCMetric raises ValueError when a window contains only one class.

    Storage uses O(window_size) pairs. Updates take O(1) time.
    Each read costs one metric construction, up to window_size metric updates,
    and one metric read. AUROCMetric and AUPRMetric sort scores, so their reads
    take O(window_size * log(window_size)) time and O(window_size) extra memory.

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
        self._pairs: deque[tuple[int, float]] = deque(maxlen=window_size)

    def update(self, y_true: int, y_pred: float) -> RollingMetric:
        """Add a label-score pair to the recent window.

        Args:
            y_true (int): The true label, 1 for an anomaly and 0 otherwise.
            y_pred (float): The predicted score or label for this instance.

        Returns:
            RollingMetric: This metric.
        """
        self._pairs.append((y_true, y_pred))
        return self

    def get(self) -> float | None:
        """Recompute the metric without changing the window.

        Returns:
            float | None: The metric result, or None for an empty window.
        """
        if not self._pairs:
            return None

        metric = self.metric_cls(**self.metric_kwargs)
        for y_true, y_pred in self._pairs:
            metric.update(y_true, y_pred)
        return metric.get()
