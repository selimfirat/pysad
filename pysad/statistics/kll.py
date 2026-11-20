from __future__ import annotations

import math

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_statistic import BaseStatistic

# Constant C = c^2 * (2c - 1) from Section 2 of Karnin et al. (2016) with c = 2/3
_C_KLL = 4.0 / 27.0


class KLLMeter(BaseStatistic):
    r"""The statistic that tracks an arbitrary quantile of the stream using the KLL sketch.

    Implements the optimal streaming quantile sketch algorithm by Karnin, Lang, and Liberty
    :cite:`karnin2016optimal`. The algorithm guarantees an :math:`\epsilon`-approximate
    rank estimate :math:`|\tilde{R}(x) - R(x)| \le \epsilon n` with probability at least
    :math:`1 - \delta`.

    The sketch combines two complementary mechanisms from the paper:

    1. **Hierarchical Compactors (Section 2)**:
       Maintains compactors across levels :math:`h`. Compactor capacities decay at rate
       :math:`c = 2/3`:

       .. math::

           k_h = \max\left(8, \left\lfloor k \cdot \left(\frac{2}{3}\right)^{H - 1 - h}\right\rfloor\right)

       When a level fills, its items are sorted and compacted in pairs using a fair coin flip
       (parity offset chosen via ``random_state``), promoting half its items to level
       :math:`h + 1` with doubled weight. If the buffer has an odd length, the remainder item is
       retained at the current level, strictly preserving the compactor's total rank mass.

    2. **Reservoir Sampler (Section 3)**:
       Maintains a sampler at height :math:`h_{\text{sampler}}` that downsamples incoming
       stream items before they enter the compactor hierarchy. When the compactor levels reach
       capacity, the lowest compactor level is retired into the sampler and
       :math:`h_{\text{sampler}}` advances, keeping the total retained items across all compactors
       and sampler strictly bounded by :math:`\mathcal{O}(k)` independent of stream length :math:`n`.
       When items in the sampler overflow capacity (:math:`v + w > 2^h`), the lighter item is kept
       with weight :math:`\min(v, w)` and the heavier item is output with weight :math:`2^h` with
       probability :math:`\max(v, w) / 2^h`.

    Merging combines another :class:`KLLMeter` with compatible capacity :math:`k`, preserving
    summary accuracy and total rank weight.

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
            for compactor parity and sampler reservoir decisions. Pass an int for reproducible
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

        # Section 2 & 3 hierarchy parameters
        self._c = 2.0 / 3.0
        self._H_max = max(2, int(math.ceil(math.log(self.k / 8.0) / math.log(1.0 / self._c))) + 1)
        self._sampler_height = 0
        self._sampler_item: float | None = None
        self._sampler_weight = 0
        self._levels: list[list[float]] = [[]]

    @property
    def num_retained_items(self) -> int:
        """The total number of items currently stored in memory across all compactors and sampler."""
        return sum(len(lvl) for lvl in self._levels) + (1 if self._sampler_weight > 0 else 0)

    @property
    def total_weight(self) -> int:
        """The total rank mass / weight preserved across all stored items."""
        tot = self._sampler_weight
        for i, lvl in enumerate(self._levels):
            h = self._sampler_height + i
            tot += len(lvl) * (1 << h)
        return tot

    @property
    def num_levels(self) -> int:
        """The number of active compactor levels currently in the sketch."""
        return len(self._levels)

    @property
    def sampler_height(self) -> int:
        """The current height h of the Section 3 reservoir sampler."""
        return self._sampler_height

    def _capacity(self, level_idx: int) -> int:
        """Computes the compactor capacity k_h for level_idx using Section 2 schedule."""
        depth = len(self._levels) - 1 - level_idx
        return max(8, int(math.floor(self.k * (self._c**depth))))

    def update(self, num: float) -> KLLMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming stream value.

        Returns:
            KLLMeter: self.
        """
        self.num_items += 1
        self._feed_sampler(float(num), 1)
        return self

    def _feed_sampler(self, item: float, weight: int) -> None:
        """Feeds an item of given weight into the Section 3 reservoir sampler."""
        if self._sampler_height == 0:
            self._levels[0].append(item)
            self._compress()
            return

        target = 1 << self._sampler_height
        v = self._sampler_weight
        w = weight

        if v + w < target:
            if v == 0:
                self._sampler_item = item
                self._sampler_weight = w
            else:
                if self._rng.uniform() < (w / (v + w)):
                    self._sampler_item = item
                self._sampler_weight = v + w
        elif v + w == target:
            if v == 0 or self._sampler_item is None:
                out_item = item
            else:
                out_item = item if self._rng.uniform() < (w / (v + w)) else self._sampler_item
            self._levels[0].append(out_item)
            self._sampler_item = None
            self._sampler_weight = 0
            self._compress()
        else:  # v + w > target (Section 3 sampler overflow rule)
            if w >= v or self._sampler_item is None:
                heavy_item, light_item = item, self._sampler_item
                heavy_w, light_w = w, v
            else:
                heavy_item, light_item = self._sampler_item, item
                heavy_w, light_w = v, w

            self._sampler_item = light_item
            self._sampler_weight = light_w

            if self._rng.uniform() < (heavy_w / target):
                self._levels[0].append(heavy_item)
                self._compress()

    def _compress(self) -> None:
        """Performs compactions across levels exceeding capacity with fair coin flips."""
        while True:
            compacted = False
            for i in range(len(self._levels)):
                cap = self._capacity(i)
                if len(self._levels[i]) >= cap:
                    compacted = True
                    buf = self._levels[i]
                    buf.sort()
                    # Preserve odd buffer remainder to maintain exact rank weight (Point 2)
                    remainder = buf.pop(-1) if len(buf) % 2 == 1 else None
                    # Fair independent coin flip (Point 1)
                    offset = int(self._rng.randint(0, 2))
                    promoted = buf[offset::2]
                    buf.clear()
                    if remainder is not None:
                        buf.append(remainder)

                    if i + 1 < len(self._levels):
                        self._levels[i + 1].extend(promoted)
                    else:
                        if len(self._levels) < self._H_max:
                            self._levels.append(promoted)
                        else:
                            # Advance sampler height to bound total levels to H_max (Section 3)
                            self._sampler_height += 1
                            old_lowest = self._levels.pop(0)
                            self._levels.append(promoted)
                            for val in old_lowest:
                                self._feed_sampler(val, 1 << (self._sampler_height - 1))
                    break
            if not compacted:
                break

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

        pairs: list[tuple[float, int]] = []
        if self._sampler_weight > 0 and self._sampler_item is not None:
            pairs.append((self._sampler_item, self._sampler_weight))

        for i, lvl in enumerate(self._levels):
            h = self._sampler_height + i
            weight = 1 << h
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

        # Align sampler heights
        while self._sampler_height < other._sampler_height:
            self._sampler_height += 1
            if len(self._levels) > 0:
                old_lowest = self._levels.pop(0)
                for val in old_lowest:
                    self._feed_sampler(val, 1 << (self._sampler_height - 1))
            if len(self._levels) == 0:
                self._levels.append([])

        # Feed other's sampler into self
        if other._sampler_weight > 0 and other._sampler_item is not None:
            self._feed_sampler(other._sampler_item, other._sampler_weight)

        # Feed other's compactors below self._sampler_height
        for i, lvl in enumerate(other._levels):
            h = other._sampler_height + i
            if h < self._sampler_height:
                w = 1 << h
                for val in lvl:
                    self._feed_sampler(val, w)
            else:
                target_idx = h - self._sampler_height
                while target_idx >= len(self._levels):
                    self._levels.append([])
                self._levels[target_idx].extend(lvl)

        self.num_items += other.num_items
        self._compress()
        return self
