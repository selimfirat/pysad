from __future__ import annotations

from pysad.core.base_statistic import UnivariateStatistic


class ExponentialMovingAverageMeter(UnivariateStatistic):
    """The statistic that tracks an exponentially weighted moving average of the stream.

    Each update moves the tracked value a fraction ``alpha`` of the way toward the new
    observation, so older observations decay geometrically and the meter keeps only its
    current value and a count. Its memory therefore stays constant as the stream grows,
    unlike a sliding window, and unlike :class:`AverageMeter` it follows a
    non-stationary baseline instead of weighting every past observation equally.

    The first observation seeds the value, so the meter answers that observation exactly
    rather than starting from zero.

    Args:
        alpha (float or None): The weight given to each new observation, which must be
            in the interval (0, 1]. Pass ``half_life`` instead to set ``alpha`` from a
            half-life in points. If neither argument is given, ``alpha`` is 0.1
            (Default=None).
        half_life (float or None): The number of points after which the weight of an
            observation halves, so ``alpha`` is ``1 - 0.5 ** (1 / half_life)``. It must
            be positive (Default=None).

    Raises:
        ValueError: If both ``alpha`` and ``half_life`` are given, if ``alpha`` is not
            in the interval (0, 1], or if ``half_life`` is not positive.

    Attributes:
        alpha (float): The weight given to each new observation.
        num_items (int): The number of items that are used to update the statistic.
    """

    def __init__(self, alpha: float | None = None, half_life: float | None = None) -> None:
        if alpha is not None and half_life is not None:
            raise ValueError("Pass only one of alpha and half_life.")

        if half_life is not None:
            if half_life <= 0:
                raise ValueError("half_life must be positive.")

            alpha = 1.0 - 0.5 ** (1.0 / half_life)

        if alpha is None:
            alpha = 0.1

        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the interval (0, 1].")

        self.alpha = alpha
        self.num_items = 0
        self._value = 0.0

    def update(self, num: float) -> ExponentialMovingAverageMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        if self.num_items == 0:
            self._value = num
        else:
            self._value = self.alpha * num + (1.0 - self.alpha) * self._value

        self.num_items += 1

        return self

    def remove(self, num: float) -> ExponentialMovingAverageMeter:
        """Raises ``NotImplementedError``, because the meter cannot remove observations.

        The meter keeps only its current value, so a past observation cannot be taken
        back. Use a windowed batch average for windows.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        raise NotImplementedError(
            "The meter keeps only its current value, so it cannot remove observations."
        )

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The current exponentially weighted moving average, or 0.0 before any update.
        """
        return self._value
