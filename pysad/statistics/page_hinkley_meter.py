from __future__ import annotations

from pysad.core.base_statistic import UnivariateStatistic


class PageHinkleyMeter(UnivariateStatistic):
    """The Page-Hinkley statistic for sequential change-point detection.

    Tracks the cumulative deviation of the stream from its running mean in
    O(1) time and memory per observation, following Page :cite:`page1954continuous`.
    Two one-sided sums are kept, one for upward shifts and one for downward shifts::

        g+_t = max(0, g+_{t-1} + x_t - mean_t - delta)
        g-_t = min(0, g-_{t-1} + x_t - mean_t + delta)

    where ``mean_t`` is the running mean including the new observation and
    ``delta`` is the smallest shift magnitude worth accumulating. By Lindley's
    identity, clamping at zero is algebraically equivalent to Page's formulation
    of subtracting the running minimum (or maximum) of the unreflected cumulative
    sum, without storing the running extrema separately. ``get`` returns
    ``max(g+, -g-)``, the two-sided statistic: a caller raises a change-point
    alarm when it exceeds a chosen threshold. Small fluctuations below ``delta``
    decay back toward zero instead of accumulating.

    Unlike the other meters, ``remove`` is not supported: the running mean and
    the cumulative sums depend on the full history, so a past observation cannot
    be taken back. This meter is meant for growing statistics.

    Args:
        delta (float): The smallest shift magnitude to accumulate, which must be
            non-negative (Default=0.0, every deviation accumulates).

    Raises:
        ValueError: If ``delta`` is negative.

    Attributes:
        delta (float): The smallest shift magnitude that is accumulated.
        num_items (int): The number of items that are used to update the statistic.
        mean (float): The running mean of the observations seen so far.
        g_plus (float): The upward cumulative sum (always non-negative).
        g_minus (float): The downward cumulative sum (always non-positive).
    """

    def __init__(self, delta: float = 0.0) -> None:
        if delta < 0.0:
            raise ValueError("delta must be non-negative.")

        self.delta = delta
        self.num_items = 0
        self.mean = 0.0
        self.g_plus = 0.0
        self.g_minus = 0.0

    def update(self, num: float) -> PageHinkleyMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        self.num_items += 1
        self.mean += (num - self.mean) / self.num_items

        deviation = num - self.mean
        self.g_plus = max(0.0, self.g_plus + deviation - self.delta)
        self.g_minus = min(0.0, self.g_minus + deviation + self.delta)

        return self

    def remove(self, num: float) -> PageHinkleyMeter:
        """Raises ``NotImplementedError``, because the Page-Hinkley state cannot remove observations.

        The running mean and the cumulative sums depend on the full history, so
        a past observation cannot be taken back. Use a windowed batch statistic
        for windows.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        raise NotImplementedError(
            "PageHinkleyMeter cannot remove observations: its state depends on the full history."
        )

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The two-sided Page-Hinkley statistic, ``max(g+, -g-)``.
        """
        return max(self.g_plus, -self.g_minus)
