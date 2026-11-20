from __future__ import annotations

from pysad.core.base_statistic import BaseStatistic
from pysad.statistics.exponential_moving_average_meter import _resolve_alpha


class ExponentialMovingVarianceMeter(BaseStatistic):
    """Track the exponentially weighted population variance of a stream.

    The first observation seeds the mean and sets the variance to zero.
    Before any update, ``get`` returns 0.0. Each later observation receives
    weight ``alpha``. Each update multiplies earlier weights by ``1 - alpha``.
    The weights sum to one. The meter does not apply a sample bias correction.
    Updates use constant time and memory. The meter does not support removal.

    Args:
        alpha (float or None): Weight for each new observation in (0, 1].
            If both options are None, use 0.1 (Default=None).
        half_life (float or None): Positive number of points that halves an
            observation's weight. Sets ``alpha = 1 - 0.5 ** (1 / half_life)``.
            Pass only one of ``alpha`` and ``half_life`` (Default=None).

    Raises:
        ValueError: If you pass both options, alpha falls outside (0, 1],
            or half_life is not positive.

    Attributes:
        alpha (float): Weight for each new observation.
        num_items (int): Number of observations.
    """

    def __init__(self, alpha: float | None = None, half_life: float | None = None) -> None:
        self.alpha = _resolve_alpha(alpha, half_life)
        self.num_items = 0
        self._mean = 0.0
        self._value = 0.0

    def update(self, num: float) -> ExponentialMovingVarianceMeter:
        """Add one observation to the statistic.

        Args:
            num (float): Incoming value.

        Returns:
            ExponentialMovingVarianceMeter: This meter.
        """
        if self.num_items == 0 or self.alpha == 1.0:
            self._mean = num
            self._value = 0.0
        else:
            delta = num - self._mean
            self._value = (1.0 - self.alpha) * (self._value + self.alpha * delta * delta)
            self._mean += self.alpha * delta

        self.num_items += 1
        return self

    def get(self) -> float:
        """Return the current population variance.

        Returns:
            float: Weighted variance, or 0.0 before any update.
        """
        return self._value
