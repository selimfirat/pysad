from __future__ import annotations

import math

from pysad.core.base_statistic import BaseStatistic


class KLLMeter(BaseStatistic):
    """The statistic that tracks a quantile of the stream using the KLL sketch.

    Implements the KLL quantile sketch algorithm by Karnin, Lang, and Liberty
    :cite:`karnin2016optimal`. The sketch maintains a hierarchy of compactors
    arranged in levels :math:`h = 0, 1, \\dots, H - 1`. Each level :math:`h`
    stores items with weight :math:`2^h`. New items arrive at level 0 with weight 1.

    When the number of items at level :math:`h` reaches its capacity :math:`k_h`,
    the level is compacted: its items are sorted and every second item is promoted
    to level :math:`h + 1`, alternating even and odd offsets across compactions
    as in Section 3 of the paper.

    The compactor capacities follow the exponential decay schedule:

    .. math::

        k_h = \\max\\left(8, \\left\\lfloor k \\cdot \\left(\\frac{2}{3}\\right)^{H - 1 - h}\\right\\rfloor\\right)

    This ensures that the total number of stored items across all levels is bounded
    by :math:`O(k)`, keeping memory strictly bounded while providing an
    :math:`\\epsilon`-approximate quantile estimate where :math:`\\epsilon \\approx 1 / k`.

    ``merge`` combines the summaries of another :class:`KLLMeter` into this meter,
    updating the compactor levels and item counts, and re-compacting if capacities
    are exceeded. The source meter remains unchanged.

    Args:
        q (float): The quantile to track, in [0, 1] (Default=0.5, the median).
        k (int): The capacity parameter bounding sketch size, :math:`k \\ge 8`
            (Default=200).
        epsilon (float | None): The target approximation error in (0, 1). If provided,
            used to set or validate accuracy; defaults to :math:`1 / k` when None.

    Raises:
        ValueError: If ``q`` is not in [0, 1], ``k < 8``, or ``epsilon`` is not in (0, 1).

    Attributes:
        q (float): The quantile tracked by default.
        k (int): The capacity parameter.
        epsilon (float): The error bound approximation parameter.
        num_items (int): The number of items observed by the meter.
    """

    def __init__(
        self,
        q: float = 0.5,
        k: int = 200,
        epsilon: float | None = None,
    ) -> None:
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be in [0, 1].")
        if k < 8:
            raise ValueError("k must be at least 8.")
        if epsilon is not None and not (0.0 < epsilon < 1.0):
            raise ValueError("epsilon must be in (0, 1).")

        self.q = float(q)
        self.k = int(k)
        self.epsilon = float(epsilon) if epsilon is not None else 1.0 / self.k
        self.num_items = 0

        self._levels: list[list[float]] = [[]]
        self._flips: list[int] = [0]

    def _capacity(self, h: int, H: int) -> int:
        """Computes the capacity for level ``h`` given total levels ``H``."""
        decay = (2.0 / 3.0) ** (H - 1 - h)
        return max(8, int(math.floor(self.k * decay)))

    def update(self, num: float) -> KLLMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value.

        Returns:
            KLLMeter: self.
        """
        self._levels[0].append(float(num))
        self.num_items += 1
        self._compress()
        return self

    def _compress(self) -> None:
        """Performs compaction across levels whose size exceeds their capacity."""
        while True:
            H = len(self._levels)
            exceeded = False
            for h in range(H):
                cap = self._capacity(h, H)
                if len(self._levels[h]) >= cap:
                    exceeded = True
                    self._levels[h].sort()
                    offset = self._flips[h] % 2
                    self._flips[h] += 1
                    promoted = self._levels[h][offset::2]
                    self._levels[h] = []

                    if h + 1 == H:
                        self._levels.append([])
                        self._flips.append(0)

                    self._levels[h + 1].extend(promoted)
                    break
            if not exceeded:
                break

    def get(self) -> float:
        """Method to obtain the tracked quantile statistic.

        Returns:
            float: The estimated quantile value.
        """
        return self.quantile(self.q)

    def quantile(self, q: float) -> float:
        """Estimates the arbitrary quantile ``q`` from the sketch.

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
        for h, buf in enumerate(self._levels):
            weight = 1 << h
            for val in buf:
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
        """
        if not isinstance(other, KLLMeter):
            raise TypeError(f"Cannot merge {type(other).__name__} into KLLMeter.")

        while len(self._levels) < len(other._levels):
            self._levels.append([])
            self._flips.append(0)

        for h, buf in enumerate(other._levels):
            self._levels[h].extend(buf)

        self.num_items += other.num_items
        self._compress()
        return self
