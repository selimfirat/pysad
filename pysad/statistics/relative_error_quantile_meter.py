from __future__ import annotations

import math

from numpy.random import RandomState
from sklearn.utils import check_random_state

from pysad.core.base_statistic import BaseStatistic


class RelativeErrorQuantileMeter(BaseStatistic):
    """Track a quantile with randomized relative rank accuracy.

    Implements Algorithms 1-3 of Cormode et al. :cite:`cormode2021relative`.
    Each level keeps a buffer of ``B = 2 * k * ceil(log2(max_items / k))``
    values. The compaction schedule counts trailing ones in its binary state.
    It compacts that many sections plus one, each with ``k`` values.
    Each compaction sorts the largest values and selects odd or even positions
    with equal probability. Selected values enter the next level with twice
    their previous weight. The smallest half of each buffer stays untouched.

    Accuracy concerns ranks, rather than the numerical distance between values.
    Larger ``k`` reduces rank error. This class does not promise a fixed error
    probability for a given ``k``. If ``high_rank_accuracy`` is true, the meter
    reverses the order and protects large ranks instead of small ranks.

    This class uses the paper's known stream limit variant from Section 2.
    It does not use the changing capacities from Section 5 or Appendix C.
    ``max_items`` bounds the total count, including merged streams.
    Memory uses O(B * log(max_items)) values and does not grow without limit.
    If ``max_items <= k``, each buffer uses one section per half.

    Unlike the authors' Python prototype, buffers keep fixed capacities.
    Each compaction draws an independent parity, as Algorithm 1 specifies.
    The prototype changes capacities and alternates parity between paired
    compactions. Both implementations combine merge states with bitwise OR.
    During merges, this class compacts buffer overflow with the scheduled tail.
    It leaves one extra value if needed to keep the compacted count even.

    ``get`` returns the first value whose cumulative weight exceeds
    ``q * (num_items - 1)``. Before compaction, this gives the exact lower
    empirical quantile. The prototype instead returns the first cumulative weight
    at least ``q * num_items``. Two extra values track the exact minimum and maximum,
    so ``q=0`` and ``q=1`` remain exact after compaction. An empty meter
    returns 0.0. The meter does not support removing observations.

    Args:
        q (float): Quantile in [0, 1] (Default=0.5).
        k (int): Positive even section size that controls accuracy (Default=50).
        max_items (int): Stream count limit, including merges (Default=2**64).
        high_rank_accuracy (bool): Protect large ranks instead of small ranks
            (Default=False).
        random_state (int, RandomState or None): Seed or random number generator.
            None uses NumPy's global random state (Default=None).

    Attributes:
        num_items (int): Number of observations, including merged observations.

    Raises:
        ValueError: If q falls outside [0, 1], k is not positive and even,
            or max_items is not positive.
    """

    def __init__(
        self,
        q: float = 0.5,
        k: int = 50,
        max_items: int = 2**64,
        high_rank_accuracy: bool = False,
        random_state: int | RandomState | None = None,
    ) -> None:
        if not 0 <= q <= 1:
            raise ValueError("q must be in [0, 1].")
        if k < 2 or k % 2:
            raise ValueError("k must be positive and even.")
        if max_items < 1:
            raise ValueError("max_items must be positive.")
        self.q = q
        self.k = k
        self.max_items = max_items
        self.high_rank_accuracy = high_rank_accuracy
        self.num_items = 0
        self._rng = check_random_state(random_state)
        self._capacity = 2 * k * max(1, ((max_items - 1) // k).bit_length())
        self._buffers: list[list[float]] = [[]]
        self._states: list[int] = [0]
        self._min = math.inf
        self._max = -math.inf

    def update(self, num: float) -> RelativeErrorQuantileMeter:
        """Add one observation.

        Args:
            num (float): Incoming value.

        Returns:
            RelativeErrorQuantileMeter: self.

        Raises:
            ValueError: If the new count exceeds max_items.
        """
        if self.num_items >= self.max_items:
            raise ValueError("The stream exceeds max_items.")
        self._insert(float(num), 0)
        self.num_items += 1
        self._min = min(self._min, num)
        self._max = max(self._max, num)
        return self

    def get(self) -> float:
        """Return the tracked quantile without changing the meter.

        Returns:
            float: Quantile estimate, or 0.0 for an empty meter.
        """
        if self.num_items == 0:
            return 0.0
        if self.q == 0:
            return float(self._min)
        if self.q == 1:
            return float(self._max)
        items = sorted(
            (value, 1 << level) for level, buffer in enumerate(self._buffers) for value in buffer
        )
        target = self.q * (self.num_items - 1)
        cumulative = 0
        for value, weight in items:
            cumulative += weight
            if cumulative > target:
                return value
        return items[-1][0]

    def merge(self, other: RelativeErrorQuantileMeter) -> RelativeErrorQuantileMeter:
        """Add another meter's observations through its weighted summary.

        The meters must share k, max_items, and high_rank_accuracy.
        The receiver keeps its quantile and random number generator.
        The source stays unchanged. If the source is self, counts double.

        Args:
            other (RelativeErrorQuantileMeter): Source meter.

        Returns:
            RelativeErrorQuantileMeter: self.

        Raises:
            ValueError: If parameters differ or the combined count exceeds max_items.
        """
        if (self.k, self.max_items, self.high_rank_accuracy) != (
            other.k,
            other.max_items,
            other.high_rank_accuracy,
        ):
            raise ValueError("Meters must share k, max_items, and high_rank_accuracy.")
        if self.num_items + other.num_items > self.max_items:
            raise ValueError("The merged stream exceeds max_items.")
        # Copy the source so self-merges read each buffer only once.
        incoming = [buffer.copy() for buffer in other._buffers]
        states = other._states.copy()
        while len(self._buffers) < len(incoming):
            self._buffers.append([])
            self._states.append(0)
        for level, buffer in enumerate(incoming):
            self._states[level] |= states[level]
            self._buffers[level].extend(buffer)
        self.num_items += other.num_items
        self._min = min(self._min, other._min)
        self._max = max(self._max, other._max)
        level = 0
        while level < len(self._buffers):
            while len(self._buffers[level]) > self._capacity:
                for value in self._compact(level):
                    self._insert(value, level + 1)
            level += 1
        return self

    def _insert(self, value: float, level: int) -> None:
        if level == len(self._buffers):
            self._buffers.append([])
            self._states.append(0)
        if len(self._buffers[level]) == self._capacity:
            for promoted in self._compact(level):
                self._insert(promoted, level + 1)
        self._buffers[level].append(value)

    def _compact(self, level: int) -> list[float]:
        buffer = self._buffers[level]
        buffer.sort(reverse=self.high_rank_accuracy)
        state = self._states[level]
        # (state XOR (state + 1)) counts trailing ones plus one through bit_length.
        sections = (state ^ (state + 1)).bit_length()
        start = self._capacity - sections * self.k
        start += (len(buffer) - start) % 2
        parity = int(self._rng.randint(2))
        promoted = buffer[start + parity :: 2]
        del buffer[start:]
        self._states[level] += 1
        return promoted
