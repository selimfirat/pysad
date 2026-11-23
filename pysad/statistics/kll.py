from __future__ import annotations

import math

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_statistic import BaseStatistic

# Constant C = c^2 * (2c - 1) from Section 2 of Karnin et al. (2016) with c = 2/3
_C_KLL = 4.0 / 27.0


class KLLMeter(BaseStatistic):
    r"""The statistic that tracks an arbitrary quantile of the stream using the KLL sketch.

    Implements the streaming quantile sketch algorithm by Karnin, Lang, and Liberty
    :cite:`karnin2016optimal` (Section 2), aligned with production reference implementations
    including Apache DataSketches Java (:class:`KllSketch`, :class:`KllHelper`) and
    `kll_sketch` Python. The algorithm guarantees an :math:`\epsilon`-approximate
    rank estimate :math:`|\tilde{R}(x) - R(x)| \le \epsilon n` with probability at least
    :math:`1 - \delta`.

    Hierarchical Compactors (Section 2):
        Maintains a hierarchy of compactor levels :math:`h \in \{0, \dots, H-1\}`.
        Each item stored at level :math:`h` represents a weight of :math:`2^h`.
        Incoming stream items are ingested at base level :math:`0` with weight :math:`1`.

    Capacity Schedule (Section 2, Theorem 1 proof):
        Level capacities decrease geometrically from the top level downwards at rate
        :math:`c = 2/3`:

        .. math::

            k_h = \max\left(8, \left\lceil k \cdot \left(\frac{2}{3}\right)^{H - 1 - h}\right\rceil\right)

        The top level (:math:`h = H - 1`) has capacity :math:`k`, and lower levels scale
        downwards bounded by a minimum compactor capacity of :math:`8` (as used in Apache
        DataSketches and ``kll_sketch``). By geometric series summation, total retained items
        across all levels are strictly bounded by :math:`\sum k_h \approx \frac{k}{1 - 2/3} = 3k`.

    Unbiased Compaction (Section 2, Figure 1; odd-length handling from Apache DataSketches):
        When a level fills, its items are sorted and compacted using two independent random
        coin flips:

        * Boundary selection (Apache DataSketches / ``kll_sketch``): When buffer length is odd,
          randomly retains either the first item (:math:`\text{index } 0`) or the last item
          (:math:`\text{index } -1`) with equal probability (:math:`0.5`), leaving an even-length
          block of remaining items. The retained boundary item stays at level :math:`h` with
          weight :math:`2^h`.
        * Pairing parity (Section 2, Figure 1 of :cite:`karnin2016optimal`): An independent random
          bit (:math:`\text{offset} \in \{0, 1\}`) determines which item of each adjacent pair is
          promoted to level :math:`h + 1` with doubled weight :math:`2^{h+1}`.

        This eliminates odd-length directional bias while strictly conserving total rank weight.

    Merge Operation (Section 2, Theorem 1 proof):
        Merging concatenates corresponding same-height compactor levels of two sketches with
        identical capacity :math:`k`, then runs compaction lazily until all levels satisfy
        their capacity constraints, preserving total rank mass and theoretical accuracy bounds.

    Args:
        q (float): The default quantile to track, in [0, 1] (Default=0.5, the median).
        k (int | None): The capacity parameter bounding sketch size, :math:`k \ge 8`.
            If None and ``epsilon`` is provided, computed from ``epsilon`` and ``delta``
            (Default=200 when ``epsilon`` is None).
        epsilon (float | None): The target rank-error approximation bound in (0, 1).
            If provided, determines or validates the required capacity :math:`k` using
            the Hoeffding failure bound :math:`\delta = 2\exp(-C \epsilon^2 k^2)`.
        delta (float): The failure probability bound in (0, 1) (Default=0.01).
        random_state (int, np.random.RandomState, or None): Controls the random coin flips
            for compactor parity and boundary retention decisions. Pass an int for reproducible
            results across runs (Default=None).

    Raises:
        ValueError: If ``q`` is not in [0, 1], ``k < 8``, ``epsilon`` is not in (0, 1),
            ``delta`` is not in (0, 1), or ``k`` is below the minimum capacity required
            by ``epsilon`` and ``delta``.

    Attributes:
        q (float): The default quantile tracked.
        k (int): The capacity parameter.
        epsilon (float): The effective rank-error bound.
        delta (float): The failure probability bound.
        num_items (int): The number of items observed in the stream.
        random_state (int, np.random.RandomState, or None): The random seed or state passed.
    """

    def __init__(
        self,
        q: float = 0.5,
        k: int | None = None,
        epsilon: float | None = None,
        delta: float = 0.01,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be in [0, 1].")
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0, 1).")

        if epsilon is not None:
            if not (0.0 < epsilon < 1.0):
                raise ValueError("epsilon must be in (0, 1).")
            min_k = int(math.ceil(math.sqrt(math.log(2.0 / delta) / _C_KLL) / epsilon))
            min_k = max(8, min_k)
            if k is not None:
                if k < 8:
                    raise ValueError("k must be at least 8.")
                if k < min_k:
                    raise ValueError(
                        f"k must be at least {min_k} to satisfy the requested epsilon={epsilon} and delta={delta}."
                    )
                resolved_k = int(k)
            else:
                resolved_k = min_k
            self.epsilon = float(epsilon)
        else:
            resolved_k = 200 if k is None else int(k)
            if resolved_k < 8:
                raise ValueError("k must be at least 8.")
            self.epsilon = math.sqrt(math.log(2.0 / delta) / _C_KLL) / resolved_k

        self.q = float(q)
        self.k = resolved_k
        self.delta = float(delta)
        self.random_state = random_state
        self._rng = check_random_state(random_state)
        self.num_items = 0

        self._c = 2.0 / 3.0
        self._levels: list[list[float]] = [[]]
        self._min_val: float | None = None
        self._max_val: float | None = None

    @property
    def num_retained_items(self) -> int:
        """The total number of items currently stored in memory across all compactors."""
        return sum(len(lvl) for lvl in self._levels)

    @property
    def total_weight(self) -> int:
        """The total rank mass / weight preserved across all stored items."""
        return sum((1 << h) * len(lvl) for h, lvl in enumerate(self._levels))

    @property
    def num_levels(self) -> int:
        """The number of active compactor levels currently in the sketch."""
        return len(self._levels)

    @property
    def sampler_height(self) -> int:
        """The height of the base compactor level (0 in pure hierarchical compaction)."""
        return 0

    def _capacity(self, level_idx: int) -> int:
        """Computes the compactor capacity k_h for level_idx using Section 2 schedule."""
        depth = len(self._levels) - 1 - level_idx
        return max(8, int(math.ceil(self.k * (self._c**depth))))

    def update(self, num: float) -> KLLMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming stream value.

        Returns:
            KLLMeter: self.
        """
        self.num_items += 1
        val = float(num)
        if self._min_val is None or val < self._min_val:
            self._min_val = val
        if self._max_val is None or val > self._max_val:
            self._max_val = val

        self._levels[0].append(val)
        self._compress()
        return self

    def _compress(self) -> None:
        """Performs hierarchical compactions on levels exceeding their capacity."""
        while True:
            compacted = False
            for i in range(len(self._levels)):
                cap = self._capacity(i)
                if len(self._levels[i]) >= cap:
                    self._compact_level(i)
                    compacted = True
                    break
            if not compacted:
                break

    def _compact_level(self, level_idx: int) -> None:
        """Compacts a single level using two independent random choices."""
        items = self._levels[level_idx]
        items.sort()

        # Random boundary retention for odd lengths (Apache DataSketches) and random parity (Section 2, Figure 1):
        # Choice 1: for odd length, randomly retain either the first item (index 0)
        # or the last item (index -1) with equal probability.
        leftover: list[float]
        compactable: list[float]
        if len(items) % 2 == 1:
            if self._rng.randint(0, 2) == 1:
                leftover = [items[-1]]
                compactable = items[:-1]
            else:
                leftover = [items[0]]
                compactable = items[1:]
        else:
            leftover = []
            compactable = items

        # Choice 2: independent parity coin flip for pairing compaction.
        offset = int(self._rng.randint(0, 2))
        promoted = compactable[offset::2]

        self._levels[level_idx] = leftover
        while len(self._levels) <= level_idx + 1:
            self._levels.append([])
        self._levels[level_idx + 1].extend(promoted)

    def get(self) -> float:
        """Method to obtain the tracked quantile statistic.

        Returns:
            float: The estimated quantile value.
        """
        return self.quantile(self.q)

    def quantile(self, q: float) -> float:
        """Estimates an arbitrary quantile ``q`` from the sketch.

        Args:
            q (float): The quantile to estimate, in [0, 1].

        Returns:
            float: The estimated value for quantile ``q``.

        Raises:
            ValueError: If ``q`` is not in [0, 1].
        """
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be in [0, 1].")

        if self.num_items == 0:
            return 0.0

        if q <= 0.0 and self._min_val is not None:
            return self._min_val
        if q >= 1.0 and self._max_val is not None:
            return self._max_val

        pairs: list[tuple[float, int]] = []
        for i, lvl in enumerate(self._levels):
            if not lvl:
                continue
            weight = 1 << i
            for val in lvl:
                pairs.append((val, weight))

        pairs.sort(key=lambda item: item[0])

        target = q * (self.num_items - 1)
        cumulative = 0
        for val, weight in pairs:
            cumulative += weight
            if cumulative > target:
                return val

        return pairs[-1][0]

    def merge(self, other: KLLMeter) -> KLLMeter:
        """Merges another KLLMeter's sketch into this meter.

        Concatenates compactor levels of the same height and lazily compacts
        any levels exceeding capacity, per Karnin et al. (2016) Section 2
        and reference implementations.

        Args:
            other (KLLMeter): The other meter to merge.

        Returns:
            KLLMeter: self.

        Raises:
            TypeError: If ``other`` is not an instance of KLLMeter.
            ValueError: If ``other.k`` does not match ``self.k``.
        """
        if not isinstance(other, KLLMeter):
            raise TypeError(f"Cannot merge {type(other).__name__} into KLLMeter.")

        if self.k != other.k:
            raise ValueError(
                f"Cannot merge KLLMeter with k={other.k} into KLLMeter with k={self.k}; capacities must match."
            )

        if other.num_items == 0:
            return self

        while len(self._levels) < len(other._levels):
            self._levels.append([])

        for idx, lvl in enumerate(other._levels):
            if lvl:
                self._levels[idx].extend(lvl)

        self.num_items += other.num_items
        if self._min_val is None or (other._min_val is not None and other._min_val < self._min_val):
            self._min_val = other._min_val
        if self._max_val is None or (other._max_val is not None and other._max_val > self._max_val):
            self._max_val = other._max_val

        self._compress()
        return self
