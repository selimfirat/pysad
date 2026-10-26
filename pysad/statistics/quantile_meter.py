from __future__ import annotations

from bisect import insort

from pysad.core.base_statistic import UnivariateStatistic


class QuantileMeter(UnivariateStatistic):
    """The statistic that tracks an arbitrary quantile of the stream in O(1) time and memory.

    Uses the P^2 algorithm of Jain & Chlamtac :cite:`jain1985psquare`, which maintains
    four interior markers at fixed quantile positions around the target quantile and
    shifts them with a parabolic predictor as observations arrive, so neither the
    observations nor their order statistics are stored.

    The first four observations are answered exactly by linear interpolation between
    the buffered order statistics. From the fifth observation on, the estimate is the
    height of the central marker, as in the paper, so the startup value and the marker
    estimate differ slightly at the switch.

    Unlike the other meters, ``remove`` is not supported: the P^2 algorithm keeps only
    its marker state, and observations cannot be removed from it. This meter is meant
    for growing statistics; windowed quantiles need a batch quantile over a window.

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

    def remove(self, num: float) -> QuantileMeter:
        """Raises ``NotImplementedError``, because the P^2 algorithm cannot remove observations.

        The meter keeps only its marker state, so a past observation cannot be taken
        back. Use a windowed batch quantile for windows.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        raise NotImplementedError(
            "The P^2 algorithm keeps only its marker state, so QuantileMeter cannot remove observations."
        )

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
