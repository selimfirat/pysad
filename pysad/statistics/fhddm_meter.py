from __future__ import annotations

import math
from collections import deque

from pysad.core.base_statistic import UnivariateStatistic


class FHDDMMeter(UnivariateStatistic):
    """The FHDDM drift detector, which compares the accuracy in a sliding window with its best value.

    Follows the Fast Hoeffding Drift Detection Method of Pesaranghader and Viktor
    :cite:`pesaranghader2016fast`. Each value is 1 for an error and 0 for a correct
    prediction. The meter keeps the last ``window_size`` values. When the window is
    full, it computes the accuracy ``mu_t`` of the window and the largest accuracy
    ``mu_m`` of a full window since the last drift. It reports a drift when::

        mu_m - mu_t >= sqrt(ln(1 / delta) / (2 * window_size))

    The right side is the Hoeffding bound ``epsilon_d`` of the paper. ``get`` returns
    the error rate of the values in the window.

    The paper inserts 1 for a correct prediction. The meter reads 1 for an error, as
    :class:`DDMMeter` does, and uses ``1 - value`` as the paper's value. The paper
    has no warning level, so the meter reports only drifts. The update after a drift
    starts a new run: it empties the window and forgets ``mu_m``, as the reset of
    the paper does. Each update takes O(1) time and O(``window_size``) memory.

    The authors' Tornado implementation and the river implementation use ``>`` in the
    test. The meter uses ``>=``, as Corollary I of the paper does. The two differ only
    when ``mu_m - mu_t`` is exactly ``epsilon_d``. River also reads 1 for an error
    but keeps the largest mean of its input, so it detects a fall of the error
    rate. River therefore matches the paper and the meter only if you pass it
    ``1 - value``.

    ``remove`` is not supported, because the detector decides when it forgets values.

    Args:
        window_size (int): The number of the last values in the window, ``n`` in the
            paper (Default=100).
        delta (float): The probability bound of the Hoeffding test, in (0, 1). Smaller
            values give fewer false drifts and later detections (Default=1e-6).

    Raises:
        ValueError: If ``delta`` is not in (0, 1).

    Attributes:
        window_size (int): The number of the last values in the window.
        delta (float): The probability bound of the Hoeffding test.
        epsilon (float): The Hoeffding bound ``epsilon_d`` of the drift test.
        num_items (int): The number of values since the last drift.
        drift_detected (bool): Whether the last update detected a drift.
    """

    def __init__(self, window_size: int = 100, delta: float = 1e-6) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1).")

        self.window_size = window_size
        self.delta = delta
        self.epsilon = math.sqrt(math.log(1.0 / delta) / (2.0 * window_size))
        self.drift_detected = False
        self._reset()

    def _reset(self) -> None:
        self.num_items = 0
        self._window: deque[float] = deque(maxlen=self.window_size)
        self._num_errors = 0.0
        # The fewest errors in a full window, which gives the largest accuracy mu_m.
        self._min_errors = math.inf

    def update(self, num: float) -> FHDDMMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, 1 for an error and 0 for a correct prediction.

        Returns:
            object: self.
        """
        if self.drift_detected:
            self.drift_detected = False
            self._reset()

        self.num_items += 1
        if len(self._window) == self.window_size:
            self._num_errors -= self._window[0]
        self._window.append(num)
        self._num_errors += num
        if self.num_items < self.window_size:
            return self

        self._min_errors = min(self._min_errors, self._num_errors)
        # mu_m - mu_t, with mu = 1 - errors / window_size.
        self.drift_detected = (
            self._num_errors - self._min_errors
        ) / self.window_size >= self.epsilon

        return self

    def remove(self, num: float) -> FHDDMMeter:
        """Raises ``NotImplementedError``, because the detector decides when it forgets values.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        raise NotImplementedError(
            "FHDDMMeter cannot remove observations: it forgets old values itself after a drift."
        )

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The error rate of the values in the window.
        """
        return self._num_errors / len(self._window)
