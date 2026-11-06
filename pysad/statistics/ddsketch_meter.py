from __future__ import annotations

import heapq
import math
import sys

from pysad.core.base_statistic import UnivariateStatistic


class DDSketchMeter(UnivariateStatistic):
    """The statistic that tracks a quantile of the stream with a relative-error guarantee.

    Uses DDSketch by Masson, Rim and Lee :cite:`masson2019ddsketch`. With
    ``gamma = (1 + alpha) / (1 - alpha)`` for the relative accuracy ``alpha``, a
    positive value ``x`` is counted in bucket ``i = ceil(log_gamma(x))``, which holds
    the values in ``(gamma ** (i - 1), gamma ** i]`` (Algorithm 1). ``get`` returns
    ``2 * gamma ** i / (gamma + 1)`` for the first bucket ``i`` whose cumulative
    count is above ``q * (n - 1)`` (Algorithm 2). The result is within a factor
    ``alpha`` of the lower ``q``-quantile of the ``n`` values.

    As in Section 2.2 of the paper, negative values are counted in a second sketch
    on their absolute values, and values within floating-point error of zero are
    counted in a special bucket for zero.

    Each of the two sketches keeps at most ``max_buckets`` buckets (Algorithm 3).
    If the positive sketch has more buckets, its lowest two buckets are merged.
    If the negative sketch has more, its highest two are merged, so the values
    closest to minus infinity are merged first, as in Section 2.2. The guarantee
    then holds if the largest value is at most ``x_q * gamma ** (max_buckets - 1)``
    (Proposition 2.6). Memory is O(max_buckets).

    The paper says only that deletion "works similarly" to insertion: ``remove``
    decrements the bucket of the value. Deviation: the paper does not say how to
    delete a value whose bucket was merged. ``remove`` then decrements the bucket
    farthest toward minus infinity, which holds the merged counts, and does not check
    that the value was added before. A bucket that a later value created below the
    merged counts can absorb such a removal, so after merges the bucket counts of
    removed values are approximate. The total count stays exact.

    Args:
        q (float): The quantile to track, in [0, 1] (Default=0.5, the median).
        relative_accuracy (float): The relative accuracy ``alpha`` of the estimate, in
            (0, 1) (Default=0.01, as in the paper's experiments).
        max_buckets (int): The largest number of buckets in each of the positive and
            negative sketches, ``m`` in the paper (Default=2048, as in the paper's
            experiments).

    Raises:
        ValueError: If ``q`` is not in [0, 1] or ``relative_accuracy`` is not in (0, 1).

    Attributes:
        q (float): The quantile that is tracked.
        relative_accuracy (float): The relative accuracy of the estimate.
        max_buckets (int): The largest number of buckets in each sketch.
        num_items (int): The number of items in the statistic.
    """

    def __init__(
        self, q: float = 0.5, relative_accuracy: float = 0.01, max_buckets: int = 2048
    ) -> None:
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be in [0, 1].")
        if not 0.0 < relative_accuracy < 1.0:
            raise ValueError("relative_accuracy must be in (0, 1).")

        self.q = q
        self.relative_accuracy = relative_accuracy
        self.max_buckets = max_buckets
        self.num_items = 0

        self._gamma = (1.0 + relative_accuracy) / (1.0 - relative_accuracy)
        self._inv_log_gamma = 1.0 / math.log(self._gamma)
        # Bucket index -> count, for the values above zero and the absolute values below zero.
        self._positive: dict[int, int] = {}
        self._negative: dict[int, int] = {}
        self._zero_count = 0

    def update(self, num: float) -> DDSketchMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        self.num_items += 1
        if abs(num) < sys.float_info.min:
            self._zero_count += 1
            return self

        store = self._positive if num > 0 else self._negative
        key = self._key(num)
        store[key] = store.get(key, 0) + 1

        if len(store) > self.max_buckets:
            # Merge the bucket farthest toward minus infinity into its neighbor.
            # Negative values are stored by absolute value, so that is their highest key.
            first, second = heapq.nsmallest(2, store) if num > 0 else heapq.nlargest(2, store)
            store[second] += store.pop(first)

        return self

    def remove(self, num: float) -> DDSketchMeter:
        """Updates the statistic by removing a particular value.

        Args:
            num (float): The value to be removed.

        Raises:
            ValueError: If no bucket can hold the value.

        Returns:
            object: self.
        """
        if abs(num) < sys.float_info.min:
            if self._zero_count == 0:
                raise ValueError(f"{num} not in DDSketchMeter")
            self._zero_count -= 1
            self.num_items -= 1
            return self

        store = self._positive if num > 0 else self._negative
        key = self._key(num)
        if key not in store and store:
            # Merged values are held by the bucket farthest toward minus infinity.
            key = max(key, min(store)) if num > 0 else min(key, max(store))

        count = store.get(key, 0)
        if count == 0:
            raise ValueError(f"{num} not in DDSketchMeter")
        if count == 1:
            del store[key]
        else:
            store[key] = count - 1
        self.num_items -= 1

        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The current quantile estimate, or 0.0 if the statistic is empty.
        """
        if self.num_items == 0:
            return 0.0

        rank = self.q * (self.num_items - 1)
        count = 0
        for key in sorted(self._negative, reverse=True):
            count += self._negative[key]
            if count > rank:
                return -self._value(key)

        count += self._zero_count
        if count > rank:
            return 0.0

        for key in sorted(self._positive):
            count += self._positive[key]
            if count > rank:
                return self._value(key)

        # Not reached: the bucket counts sum to num_items, which is above rank.
        return self._value(max(self._positive))

    def _key(self, num: float) -> int:
        return math.ceil(math.log(abs(num)) * self._inv_log_gamma)

    def _value(self, key: int) -> float:
        return 2.0 * self._gamma**key / (self._gamma + 1.0)
