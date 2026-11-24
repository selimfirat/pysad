from __future__ import annotations

import math

from pysad.core.base_statistic import BaseStatistic


class GeometricMovingAverageMeter(BaseStatistic):
    """The geometric moving average of Roberts (1959) with a drift alarm.

    Tracks the geometric moving average of non-negative observations. Each new
    observation X_t updates the weighted sum of log observations and the weight
    sum::

        A = alpha * log(X_t) + (1 - alpha) * A
        W = alpha + (1 - alpha) * W
        geometric_mean = exp(A / W)

    The weight of the newest observation is ``alpha`` and older observations
    decay geometrically, so the meter keeps constant memory as the stream grows.
    This is the geometric moving average control chart statistic described by
    Roberts (1959), "Control chart tests based on geometric moving averages",
    Technometrics, 1(3), 239-250. The first observation seeds the average exactly.

    Because every past observation keeps a non-zero weight under exponential
    decay, a zero observation drives the geometric moving average to zero. The
    meter therefore accepts zero and reports a geometric mean of zero from that
    point on; negative observations are rejected.

    A drift is reported when the new observation differs from the current
    geometric moving average by more than ``threshold`` on the absolute log
    scale, that is when ``abs(log(X_t) - log(geometric_mean_{t-1})) > threshold``.
    This is a simple control-chart alarm; calibrate ``threshold`` to the scale of
    the data.

    Each update takes O(1) time and memory. The meter does not support
    ``remove``.

    Args:
        alpha (float): The weight of the newest observation, in the interval
            (0, 1] (Default=0.1).
        threshold (float): The drift alarm threshold on the absolute log-scale
            deviation. It must be non-negative (Default=1.0).

    Raises:
        ValueError: If ``alpha`` is not in (0, 1], if ``threshold`` is
            negative, or if a negative observation is passed to ``update``.

    Attributes:
        alpha (float): The weight of the newest observation.
        threshold (float): The drift alarm threshold.
        num_items (int): The number of observations seen.
        drift_detected (bool): Whether the last update reported a drift.
    """

    def __init__(self, alpha: float = 0.1, threshold: float = 1.0) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the interval (0, 1].")
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative.")

        self.alpha = alpha
        self.threshold = threshold
        self.num_items = 0
        self.drift_detected = False
        self._A = 0.0
        self._W = 0.0
        self._geometric_mean = 0.0

    def update(self, num: float) -> GeometricMovingAverageMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, which must be non-negative.

        Returns:
            object: self.
        """
        if num < 0.0:
            raise ValueError("observations must be non-negative.")

        log_num = math.log(num) if num > 0.0 else -math.inf
        if self.num_items == 0:
            # Seed so the first observation is the average exactly.
            self._A = self.alpha * log_num
            self._W = self.alpha
            self._geometric_mean = num
            self.drift_detected = False
        else:
            prev_log_mean = self._A / self._W
            deviation = abs(log_num - prev_log_mean)
            self.drift_detected = deviation > self.threshold
            keep = 1.0 - self.alpha
            self._A = self.alpha * log_num + keep * self._A
            self._W = self.alpha + keep * self._W
            self._geometric_mean = math.exp(self._A / self._W)

        self.num_items += 1
        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The current geometric moving average, or 0.0 before any update.
        """
        return self._geometric_mean
