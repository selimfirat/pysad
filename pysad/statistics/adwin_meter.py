from __future__ import annotations

import math

from pysad.core.base_statistic import UnivariateStatistic


class ADWINMeter(UnivariateStatistic):
    """The mean of an adaptive window that shrinks when the stream changes.

    Follows ADWIN2 by Bifet and Gavaldà :cite:`bifet2007learning`. The window
    ``W`` holds the most recent items and grows while the stream is stationary.
    After each update, every split of ``W`` into an older part ``W0`` and a newer
    part ``W1`` at a bucket border is tested, and the oldest bucket is dropped
    while some split has ``|mean(W0) - mean(W1)| > eps_cut``, with the paper's
    normal-approximation bound (Eq. 3.1)::

        eps_cut = sqrt(2 / m * var(W) * ln(2 / delta')) + 2 / (3 * m) * ln(2 / delta')

    where ``1 / m = 1 / n0 + 1 / n1`` for the split lengths ``n0`` and ``n1``,
    ``var(W)`` is the variance of the window, and ``delta' = delta / ln(n)`` for
    the window length ``n``. ``get`` returns the mean of the window.

    The window is kept as an exponential histogram: buckets whose sizes are powers
    of two, with at most ``max_buckets`` buckets of each size. When a size has
    ``max_buckets + 1`` buckets, its two oldest are merged into one bucket of the
    next size, so memory and the number of tested splits are O(log W).

    Deviation from the paper: each bucket also keeps the sum of squared deviations
    of its items, so ``var(W)`` is exact after buckets are merged or dropped. The
    paper stores only the size and the sum of each bucket and does not say how
    ``var(W)`` is maintained.

    ``remove`` is not supported, because ADWIN decides which items leave its window.

    Args:
        delta (float): The confidence of the change test, in (0, 1). Smaller values
            give fewer false alarms and slower detection (Default=0.002).
        max_buckets (int): The largest number of buckets of each size, ``M`` in the
            paper (Default=5).

    Raises:
        ValueError: If ``delta`` is not in (0, 1).

    Attributes:
        delta (float): The confidence of the change test.
        max_buckets (int): The largest number of buckets of each size.
        width (int): The number of items in the window.
        drift_detected (bool): Whether the last update shrank the window.
    """

    def __init__(self, delta: float = 0.002, max_buckets: int = 5) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1).")

        self.delta = delta
        self.max_buckets = max_buckets
        self.width = 0
        self.drift_detected = False
        self._sum = 0.0
        self._sq_dev = 0.0  # Sum of squared deviations from the window mean.
        # _buckets[i] holds the [sum, sum of squared deviations] of each bucket of 2**i items, oldest first.
        self._buckets: list[list[list[float]]] = []

    def update(self, num: float) -> ADWINMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        if self.width > 0:
            diff = num - self._sum / self.width
            self._sq_dev += diff * diff * self.width / (self.width + 1)
        self._sum += num
        self.width += 1

        if not self._buckets:
            self._buckets.append([])
        self._buckets[0].append([num, 0.0])
        self._compress()

        self.drift_detected = False
        while self._has_cut():
            self._drop_oldest()
            self.drift_detected = True

        return self

    def remove(self, num: float) -> ADWINMeter:
        """Raises ``NotImplementedError``, because ADWIN decides which items leave its window.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        raise NotImplementedError(
            "ADWINMeter cannot remove observations: it drops old items itself when the stream changes."
        )

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The mean of the window.
        """
        return self._sum / self.width

    def _compress(self) -> None:
        for level, buckets in enumerate(self._buckets):
            if len(buckets) <= self.max_buckets:
                return

            size = 2**level
            (sum0, sq_dev0), (sum1, sq_dev1) = buckets.pop(0), buckets.pop(0)
            merged = [sum0 + sum1, sq_dev0 + sq_dev1 + (sum0 - sum1) ** 2 / (2 * size)]
            if level + 1 == len(self._buckets):
                self._buckets.append([])
            self._buckets[level + 1].append(merged)

    def _has_cut(self) -> bool:
        n = self.width
        if n < 2:
            return False

        variance = self._sq_dev / n
        log_term = math.log(2.0 * math.log(n) / self.delta)
        n0, sum0 = 0, 0.0
        for level in range(len(self._buckets) - 1, -1, -1):
            for bucket_sum, _ in self._buckets[level]:
                n0 += 2**level
                sum0 += bucket_sum
                n1 = n - n0
                if n1 == 0:
                    return False

                inv_m = 1.0 / n0 + 1.0 / n1
                eps_cut = (
                    math.sqrt(2.0 * inv_m * variance * log_term) + 2.0 / 3.0 * inv_m * log_term
                )
                if abs(sum0 / n0 - (self._sum - sum0) / n1) > eps_cut:
                    return True

        return False

    def _drop_oldest(self) -> None:
        size = 2 ** (len(self._buckets) - 1)
        bucket_sum, bucket_sq_dev = self._buckets[-1].pop(0)
        rest = self.width - size
        diff = bucket_sum / size - (self._sum - bucket_sum) / rest
        self._sq_dev = max(
            0.0, self._sq_dev - bucket_sq_dev - diff * diff * size * rest / self.width
        )
        self._sum -= bucket_sum
        self.width = rest
        while not self._buckets[-1]:
            self._buckets.pop()
