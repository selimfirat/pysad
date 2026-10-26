from __future__ import annotations

from pysad.core.base_statistic import UnivariateStatistic


class QuantileMeter(UnivariateStatistic):
    """The statistic that estimates a streaming quantile in O(1) memory with the P² algorithm :cite:`jain1985p`.

    The meter keeps the five markers of the P² algorithm: the minimum, the p/2-, p- and (1+p)/2-quantile
    estimates, and the maximum, together with their integer positions in the stream and their desired
    positions. Each incoming value moves every marker behind it one position up, and the three middle
    markers drift toward their desired positions by a piecewise-parabolic prediction that falls back to a
    linear one whenever the prediction would leave the interval of the neighboring markers, as described
    in the paper. The estimate is the middle marker.

    The P² markers are an estimate, not exact bookkeeping, so :meth:`remove` is not supported and the
    meter cannot be wrapped by :class:`~pysad.statistics.RunningStatistic`. Until the fifth observation
    arrives, ``get`` returns the exact quantile of the observations so far, and 0.0 before the first one.

    Args:
        q (float): The quantile to track, strictly between 0 and 1 (Default=0.5).

    Raises:
        ValueError: If `q` is not strictly between 0 and 1.

    Attributes:
        q (float): The quantile that is estimated.
        num_items (int): The number of items that are used to update the statistic.
    """

    def __init__(self, q: float = 0.5) -> None:
        if not 0 < q < 1:
            raise ValueError(f"q must be in (0, 1), got {q!r}.")

        self.q = float(q)
        self.num_items = 0

        # The first five observations initialize the marker heights.
        self._initial: list[float] = []
        # Marker heights q_i, integer marker positions n_i, and desired positions n'_i of the
        # paper, zero-based. The desired-position increments dn'_i are folded into _increments.
        self._heights: list[float] = [0.0] * 5
        self._positions = [0] * 5
        self._desired = [0.0] * 5
        self._increments = [0.0, q / 2.0, q, (1.0 + q) / 2.0, 1.0]

    def update(self, num: float) -> QuantileMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        self.num_items += 1

        if self.num_items <= 5:
            self._initial.append(float(num))
            if self.num_items < 5:
                return self

            # The paper's initialization: the sorted first five observations become the marker
            # heights at positions 0 to 4, and the desired positions are (n - 1) * p_i for n = 5.
            heights, desired, positions = self._heights, self._desired, self._positions
            self._initial.sort()
            heights[:] = self._initial
            positions[:] = range(5)
            desired[1], desired[2], desired[3], desired[4] = (
                2.0 * self.q,
                4.0 * self.q,
                2.0 + 2.0 * self.q,
                4.0,
            )
            self._initial = []
            return self

        heights, positions, desired = self._heights, self._positions, self._desired

        # Every marker behind the cell the value falls into moves one position up; a value outside
        # the current extremes also updates the outermost marker height.
        if num < heights[0]:
            heights[0] = num
            k = 0
        elif num < heights[1]:
            k = 0
        elif num < heights[2]:
            k = 1
        elif num < heights[3]:
            k = 2
        elif num < heights[4]:
            k = 3
        else:
            heights[4] = num
            k = 3

        for i in range(k + 1, 5):
            positions[i] += 1
        for i in range(5):
            desired[i] += self._increments[i]

        # Adjust the middle markers toward their desired positions, one step at a time, and only
        # when the neighboring marker is more than one position away, as in the paper's Figure 5.
        for i in range(1, 4):
            d = desired[i] - positions[i]
            if (d >= 1 and positions[i + 1] - positions[i] > 1) or (
                d <= -1 and positions[i - 1] - positions[i] < -1
            ):
                step = 1 if d > 0 else -1
                below, above = positions[i - 1], positions[i + 1]
                left, middle, right = heights[i - 1], heights[i], heights[i + 1]
                gap_below, gap_above = middle - left, right - middle

                # Piecewise-parabolic prediction, and the linear one when it leaves the interval
                # of the neighboring markers.
                predicted = middle + step / (above - below) * (
                    (positions[i] - below + step) * gap_above / (above - positions[i])
                    + (above - positions[i] - step) * gap_below / (positions[i] - below)
                )
                if not left < predicted < right:
                    neighbor = i + step
                    predicted = middle + step * (heights[neighbor] - middle) / (
                        positions[neighbor] - positions[i]
                    )

                heights[i] = predicted
                positions[i] += step

        return self

    def remove(self, num: float) -> QuantileMeter:
        """Not supported: the P² markers estimate the stream, so a value cannot be taken back out.

        Args:
            num (float): Ignored; the removal is always refused.

        Returns:
            object: self. This method never returns normally.

        Raises:
            ValueError: Always, because the estimate cannot absorb a removal. The meter therefore
                cannot be wrapped by :class:`~pysad.statistics.RunningStatistic`.
        """
        raise ValueError(
            "QuantileMeter estimates the markers of the P² algorithm, so it cannot remove a value. "
            "Use it on its own rather than inside RunningStatistic."
        )

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The current quantile estimate. Before the fifth observation, the exact quantile
            of the observations so far (0.0 when there are none).
        """
        if self.num_items == 0:
            return 0.0

        if len(self._initial) > 0:
            ordered = sorted(self._initial)
            # Linear interpolation between the closest ranks, as numpy's default quantile method.
            rank = (len(ordered) - 1) * self.q
            low = int(rank)
            if low == rank:
                return ordered[low]
            return ordered[low] + (rank - low) * (ordered[low + 1] - ordered[low])

        return self._heights[2]
