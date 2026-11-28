from __future__ import annotations

import math

from pysad.core.base_statistic import BaseStatistic

__bibtex__ = r"""
@article{roberts1959control,
  title={Control chart tests based on geometric moving averages},
  author={Roberts, S. W.},
  journal={Technometrics},
  volume={1},
  number={3},
  pages={239--250},
  year={1959},
  publisher={Taylor \& Francis}
}
"""


class EWMAControlChartMeter(BaseStatistic):
    """Flag a shift in the level of a stream with an EWMA control chart.

    Follows the geometric moving average control chart of Roberts
    :cite:`roberts1959control`. The chart starts at the target ``mean`` and
    each update weighs the new value by ``alpha`` (``r`` in the paper)::

        z_t = alpha * x_t + (1 - alpha) * z_{t-1},  z_0 = mean
        limit_t = threshold * std * sqrt(alpha / (2 - alpha) * (1 - (1 - alpha) ** (2 * t)))

    ``limit_t`` is ``threshold`` times the standard deviation of ``z_t`` for
    independent values with standard deviation ``std``, so the limits widen
    over the first updates towards their steady-state value. ``get`` returns
    ``z_t``. If ``abs(z_t - mean)`` exceeds ``limit_t``, the meter sets
    ``drift_detected`` to True. Equality does not trigger an alarm.

    Like :class:`CUSUMMeter`, the meter takes ``mean`` and ``std`` as fixed
    in-control values; set them from a stable reference stream. The chart
    keeps running after an alarm, as in the paper. Each update takes O(1)
    time and memory. The meter does not support ``remove``.

    Args:
        mean (float): The fixed in-control mean (Default=0.0).
        std (float): The positive in-control standard deviation (Default=1.0).
        alpha (float): The weight of the newest value, in (0, 1] (Default=0.2).
        threshold (float): The positive width of the control limits, in
            standard deviations of ``z_t`` (Default=3.0).

    Raises:
        ValueError: If ``std`` or ``threshold`` is not positive, or if
            ``alpha`` is not in (0, 1].

    Attributes:
        limit_ (float): The current half-width of the control limits, or 0.0
            before the first update.
        drift_detected (bool): Whether the last update fell outside the limits.
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        alpha: float = 0.2,
        threshold: float = 3.0,
    ) -> None:
        if std <= 0.0:
            raise ValueError("std must be positive.")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the interval (0, 1].")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive.")

        self.mean = mean
        self.std = std
        self.alpha = alpha
        self.threshold = threshold
        self.limit_ = 0.0
        self.drift_detected = False
        self._ewma = mean
        # (1 - alpha) ** (2 * t), the part of the steady-state variance not yet reached.
        self._decay = 1.0

    def update(self, num: float) -> EWMAControlChartMeter:
        """Add a value and update the drift flag.

        Args:
            num (float): The next value in the stream.

        Returns:
            EWMAControlChartMeter: self.
        """
        self._ewma = self.alpha * num + (1.0 - self.alpha) * self._ewma
        self._decay *= (1.0 - self.alpha) ** 2
        variance = self.alpha / (2.0 - self.alpha) * (1.0 - self._decay)
        self.limit_ = self.threshold * self.std * math.sqrt(variance)
        self.drift_detected = abs(self._ewma - self.mean) > self.limit_
        return self

    def get(self) -> float:
        """Return the exponentially weighted moving average without changing state.

        Returns:
            float: The current EWMA, or ``mean`` before the first update.
        """
        return self._ewma
