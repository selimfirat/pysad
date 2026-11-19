from __future__ import annotations

import heapq
import math
import sys
from bisect import bisect_left, insort

from pysad.core.base_statistic import BaseStatistic, RemovableStatistic


class DDSketchMeter(RemovableStatistic):
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
    (Proposition 4). Memory is O(max_buckets).

    ``merge`` combines bucket counts from another meter with the same relative
    accuracy and collapses buckets to this meter's limit (Algorithm 4). It keeps
    this meter's quantile and leaves the other meter unchanged. Collapsed buckets
    cannot be recovered by merging into a meter with a larger limit, so quantiles
    in those buckets may still fall outside the relative-error guarantee.

    Without bucket collapses, the estimates equal those of the authors' Python implementation
    ``ddsketch`` (sketches-py) with the logarithmic mapping. With collapses they can
    differ: sketches-py bounds the range of bucket indices, and its negative store
    merges the values closest to zero, but the paper bounds the number of non-empty
    buckets, and this class follows the paper.

    The paper says only that deletion "works similarly" to insertion: ``remove``
    decrements the bucket of the value. Deviation: the paper does not say how to
    delete a value whose bucket was merged. A merge moves counts toward plus
    infinity, so ``remove`` then decrements the nearest existing bucket toward plus
    infinity. This bucket can be one that a later value created between the merged
    bucket and the bucket that holds the merged counts, so after merges the bucket
    counts of removed values are approximate. The total count stays exact, and the
    removal of a value that was added and not yet removed never raises an error.

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

    def merge(self, other: DDSketchMeter) -> DDSketchMeter:
        """Adds another meter's observations, preserving this meter's bucket limit.

        Uses Algorithm 4 of :cite:`masson2019ddsketch`, with negative buckets
        collapsed from the highest index as in Section 2.2. Work and temporary
        storage depend on the bucket limits, not the number of observations.

        Args:
            other (DDSketchMeter): Meter with the same relative accuracy. Its
                quantile and bucket limit may differ. It is left unchanged unless
                it is this meter itself, in which case all counts are doubled.

        Raises:
            ValueError: If the meters have different relative accuracies.

        Returns:
            DDSketchMeter: self.
        """
        if self.relative_accuracy != other.relative_accuracy:
            raise ValueError("Meters must have the same relative_accuracy to merge.")

        for store, incoming, reverse in (
            (self._positive, other._positive, False),
            (self._negative, other._negative, True),
        ):
            for key, count in incoming.items():
                store[key] = store.get(key, 0) + count

            if len(store) > self.max_buckets:
                keys = sorted(store, reverse=reverse)
                cutoff = len(keys) - self.max_buckets
                # Collapsing adjacent buckets repeatedly adds all removed counts here.
                store[keys[cutoff]] += sum(store.pop(key) for key in keys[:cutoff])

        self._zero_count += other._zero_count
        self.num_items += other.num_items
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
        if key not in store:
            # A merge moved the counts of this bucket to the next bucket toward plus infinity.
            # Negative values are stored by absolute value, so that is a lower key.
            if num > 0:
                key = min((k for k in store if k > key), default=key)
            else:
                key = max((k for k in store if k < key), default=key)

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


class MedianMeter(RemovableStatistic):
    """The statistic that keeps track of the median.

    Uses a sorted list with :func:`bisect.insort` when tracking 1,000 items or fewer,
    and switches to two heaps with lazy deletion for windows exceeding 1,000 items
    to bound per-instance latency to :math:`O(\\log W)`.

    Args:
        high_threshold (int): The maximum number of items to track via sorted list
            before transitioning to two heaps (Default=1000).
        low_threshold (int): The threshold below which two heaps are converted
            back to a sorted list (Default=800).

    Attributes:
        num_items (int): The number of items that are used to update the statistic.
        lst (list[float]): The list of values that are used to update the statistic
            when in list mode.
    """

    def __init__(self, high_threshold: int = 1000, low_threshold: int = 800) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.num_items = 0
        self.is_heap = False

        self.lst: list[float] = []
        self.small: list[float] = []
        self.large: list[float] = []
        self.small_size = 0
        self.large_size = 0
        self.delayed: dict[float, int] = {}
        self.counts: dict[float, int] = {}

    def _prune(self, heap: list[float], is_small: bool) -> None:
        sign = -1.0 if is_small else 1.0
        while heap:
            val = sign * heap[0]
            cnt = self.delayed.get(val, 0)
            if cnt > 0:
                if cnt == 1:
                    del self.delayed[val]
                else:
                    self.delayed[val] = cnt - 1
                heapq.heappop(heap)
            else:
                break

    def _balance(self) -> None:
        if self.small_size > self.large_size + 1:
            self._prune(self.small, is_small=True)
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
            self.small_size -= 1
            self.large_size += 1
            self._prune(self.small, is_small=True)
        elif self.small_size < self.large_size:
            self._prune(self.large, is_small=False)
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)
            self.large_size -= 1
            self.small_size += 1
            self._prune(self.large, is_small=False)

    def _to_heaps(self) -> None:
        for x in self.lst:
            self.counts[x] = self.counts.get(x, 0) + 1
        mid = (len(self.lst) + 1) // 2
        self.small = [-x for x in self.lst[:mid]]
        self.small_size = len(self.small)
        heapq.heapify(self.small)
        self.large = list(self.lst[mid:])
        self.large_size = len(self.large)
        heapq.heapify(self.large)
        self.lst.clear()
        self.is_heap = True
        self._balance()

    def _to_list(self) -> None:
        lst: list[float] = []
        for val, c in self.counts.items():
            lst.extend([val] * c)
        lst.sort()
        self.lst = lst
        self.small.clear()
        self.large.clear()
        self.delayed.clear()
        self.counts.clear()
        self.small_size = 0
        self.large_size = 0
        self.is_heap = False

    def update(self, num: float) -> MedianMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        if not self.is_heap:
            if self.num_items < self.high_threshold:
                insort(self.lst, num)
                self.num_items += 1
                return self
            self._to_heaps()

        self.counts[num] = self.counts.get(num, 0) + 1
        self._prune(self.small, is_small=True)
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
            self.small_size += 1
        else:
            heapq.heappush(self.large, num)
            self.large_size += 1

        self._balance()
        self.num_items += 1

        return self

    def remove(self, num: float) -> MedianMeter:
        """Updates the statistic by removing a particular value.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        if not self.is_heap:
            idx = bisect_left(self.lst, num)
            if idx < self.num_items and self.lst[idx] == num:
                del self.lst[idx]
                self.num_items -= 1
                return self
            raise ValueError(f"{num} not in MedianMeter")

        cnt = self.counts.get(num, 0)
        if cnt <= 0:
            raise ValueError(f"{num} not in MedianMeter")
        if cnt == 1:
            del self.counts[num]
        else:
            self.counts[num] = cnt - 1

        self._prune(self.small, is_small=True)
        if self.small and num <= -self.small[0]:
            self.small_size -= 1
            self.delayed[num] = self.delayed.get(num, 0) + 1
            if num == -self.small[0]:
                self._prune(self.small, is_small=True)
        else:
            self.large_size -= 1
            self.delayed[num] = self.delayed.get(num, 0) + 1
            self._prune(self.large, is_small=False)

        self._balance()
        self.num_items -= 1

        if self.num_items < self.low_threshold:
            self._to_list()

        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The statistic.
        """
        if self.num_items == 0:
            return 0.0

        if not self.is_heap:
            if self.num_items % 2 == 0:
                return (self.lst[self.num_items // 2] + self.lst[self.num_items // 2 - 1]) / 2
            return self.lst[self.num_items // 2]

        self._prune(self.small, is_small=True)
        if self.small_size == self.large_size:
            self._prune(self.large, is_small=False)
            return (-self.small[0] + self.large[0]) / 2.0
        return float(-self.small[0])


class QuantileMeter(BaseStatistic):
    """The statistic that tracks an arbitrary quantile of the stream in O(1) time and memory.

    Uses the P^2 algorithm of Jain & Chlamtac :cite:`jain1985psquare`, which maintains
    four interior markers at fixed quantile positions around the target quantile and
    shifts them with a parabolic predictor as observations arrive, so neither the
    observations nor their order statistics are stored.

    The first four observations are answered exactly by linear interpolation between
    the buffered order statistics. From the fifth observation on, the estimate is the
    height of the central marker, as in the paper, so the startup value and the marker
    estimate differ slightly at the switch.

    ``remove`` is not supported: the P^2 algorithm keeps only its marker state, and
    observations cannot be removed from it. Use :class:`DDSketchMeter` or
    :class:`MedianMeter` with :class:`RunningStatistic` for windowed quantiles.

    Args:
        q (float): The quantile to track, which must be in the open interval (0, 1)
            (Default=0.5, the median).

    Raises:
        ValueError: If ``q`` is not in the open interval (0, 1).

    Attributes:
        q (float): The quantile that is tracked.
        num_items (int): The number of items that are used to update the statistic.
    """

    def __init__(self, q: float = 0.5) -> None:
        if not 0.0 < q < 1.0:
            raise ValueError("q must be in the open interval (0, 1).")

        self.q = q
        self.num_items = 0

        # The sorted sample the meter starts from, and the markers it initializes
        # from it. The paper starts the algorithm from the fifth observation.
        self._sample: list[float] = []
        self._heights: list[float] = []
        self._positions: list[int] = []
        self._desired: list[float] = []
        self._increments: list[float] = []

    def update(self, num: float) -> QuantileMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        self.num_items += 1

        if self.num_items <= 5:
            insort(self._sample, num)

            if self.num_items == 5:
                self._initialize_markers()

            return self

        heights = self._heights
        positions = self._positions
        desired = self._desired

        # Step 1: advance the desired marker positions.
        increments = self._increments
        for i in range(5):
            desired[i] += increments[i]

        # Step 2: place the observation in one of the four marker intervals, growing
        # the extreme markers when the observation falls outside them.
        if num < heights[0]:
            heights[0] = num
            cell = 0
        elif num >= heights[4]:
            heights[4] = num
            cell = 3
        else:
            cell = next((i for i in range(4) if heights[i] <= num < heights[i + 1]), 3)

        # Step 3: shift the positions of the markers above the new observation.
        for i in range(cell + 1, 5):
            positions[i] += 1

        # Step 4: move the interior markers toward their desired positions, one
        # position at a time, as in the paper.
        for i in range(1, 4):
            d = desired[i] - positions[i]
            if (d >= 1 and positions[i + 1] - positions[i] > 1) or (
                d <= -1 and positions[i - 1] - positions[i] < -1
            ):
                d = 1 if d > 0 else -1
                heights[i] = self._predicted_height(i, d)
                positions[i] += d

        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The current quantile estimate.
        """
        if self.num_items == 0:
            return 0.0

        if self.num_items < 5:
            return self._exact_quantile(self._sample)

        return self._heights[2]

    def _initialize_markers(self) -> None:
        """Initializes the markers from the first five sorted observations, as in the paper."""
        p = self.q
        self._heights = list(self._sample)
        self._positions = [1, 2, 3, 4, 5]
        self._desired = [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0]
        self._increments = [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0]

    def _predicted_height(self, i: int, d: int) -> float:
        """Predicts marker ``i``'s new height with the parabolic predictor, or the linear one.

        The paper falls back to the linear predictor whenever the parabolic prediction
        leaves the interval of the neighboring marker heights.
        """
        heights = self._heights
        positions = self._positions

        parabolic = heights[i] + (d / (positions[i + 1] - positions[i - 1])) * (
            (positions[i] - positions[i - 1] + d)
            * (heights[i + 1] - heights[i])
            / (positions[i + 1] - positions[i])
            + (positions[i + 1] - positions[i] - d)
            * (heights[i] - heights[i - 1])
            / (positions[i] - positions[i - 1])
        )

        if heights[i - 1] < parabolic < heights[i + 1]:
            return parabolic

        # d is +1 or -1 here, so i + d is the neighbor in the direction of the move.
        return heights[i] + d * (heights[i + d] - heights[i]) / (positions[i + d] - positions[i])

    def _exact_quantile(self, values: list[float]) -> float:
        """Interpolates the exact quantile of the buffered startup sample."""
        h = (len(values) - 1) * self.q
        lower = int(h)
        fraction = h - lower

        if fraction == 0.0:
            return float(values[lower])

        return float(values[lower] + fraction * (values[lower + 1] - values[lower]))
