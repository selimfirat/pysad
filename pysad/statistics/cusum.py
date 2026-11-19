from __future__ import annotations

from pysad.core.base_statistic import BaseStatistic


class CUSUMMeter(BaseStatistic):
    """Track shifts above and below a fixed target mean with Page's CUSUM.

    Follows the cumulative sum inspection scheme of Page
    :cite:`page1954continuous`. Each update adds the deviation from ``mean``
    to two sums, with an allowance ``delta``::

        g_plus = max(0, g_plus + x - mean - delta)
        g_minus = min(0, g_minus + x - mean + delta)

    ``get`` returns ``max(g_plus, -g_minus)``. If this value exceeds
    ``threshold``, the meter sets ``drift_detected`` to True. Equality does
    not trigger an alarm. The next update resets both sums before it adds
    the new value. The target mean stays fixed after an alarm.

    Unlike :class:`PageHinkleyMeter`, this meter does not estimate a running
    mean. Set ``mean`` from a stable reference stream. All parameters use
    the input units. For standardized inputs, the defaults target a shift
    of one standard deviation.

    API choice: the meter keeps the alarm value until the next update,
    rather than stopping inspection for corrective action. Each update
    takes O(1) time and memory. The meter does not support ``remove``.

    The R package qcc uses the same sums with a negative lower sum.
    With unit standard deviation, its ``center``, ``se.shift / 2``, and
    ``decision.interval`` match ``mean``, ``delta``, and ``threshold``.
    Unlike qcc, this meter restarts after an alarm and stores no history.

    Args:
        mean (float): The fixed target mean (Default=0.0).
        delta (float): The non-negative allowance for each deviation
            (Default=0.5).
        threshold (float): The positive alarm limit for either sum
            (Default=5.0).

    Raises:
        ValueError: If ``delta`` is negative or ``threshold`` is not positive.

    Attributes:
        g_plus_ (float): The upward cumulative deviation, at least zero.
        g_minus_ (float): The downward cumulative deviation, at most zero.
        drift_detected (bool): Whether the last update detected a shift.
    """

    def __init__(
        self,
        mean: float = 0.0,
        delta: float = 0.5,
        threshold: float = 5.0,
    ) -> None:
        if delta < 0.0:
            raise ValueError("delta must be non-negative.")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive.")

        self.mean = mean
        self.delta = delta
        self.threshold = threshold
        self.g_plus_ = 0.0
        self.g_minus_ = 0.0
        self.drift_detected = False

    def update(self, num: float) -> CUSUMMeter:
        """Add a value and update the drift flag.

        Args:
            num (float): The next value in the stream.

        Returns:
            CUSUMMeter: self.
        """
        if self.drift_detected:
            self.g_plus_ = self.g_minus_ = 0.0

        deviation = num - self.mean
        self.g_plus_ = max(0.0, self.g_plus_ + deviation - self.delta)
        self.g_minus_ = min(0.0, self.g_minus_ + deviation + self.delta)
        self.drift_detected = self.get() > self.threshold
        return self

    def get(self) -> float:
        """Return the larger cumulative deviation without changing state.

        Returns:
            float: The two-sided statistic, or 0.0 before the first update.
        """
        return max(self.g_plus_, -self.g_minus_)
