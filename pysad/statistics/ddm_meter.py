from __future__ import annotations

import math

from pysad.core.base_statistic import UnivariateStatistic


class DDMMeter(UnivariateStatistic):
    """The DDM drift detector, which watches the error rate of a stream of 0/1 errors.

    Follows the Drift Detection Method of Gama et al. :cite:`gama2004learning`
    (Section 3). Each value is 1 for an error and 0 for a correct prediction. After
    ``i`` values, the meter keeps the error rate ``p_i`` and its standard deviation
    ``s_i = sqrt(p_i * (1 - p_i) / i)``. It records ``p_min`` and ``s_min`` when
    ``p_i + s_i`` is lower than ``p_min + s_min``. It reports a warning when
    ``p_i + s_i > p_min + warning_level * s_min`` and a drift when
    ``p_i + s_i > p_min + drift_level * s_min``. ``get`` returns ``p_i``.

    As in the paper, the normal approximation and the tests apply only after more
    than ``min_num_items`` values. The update after a drift starts a new context: it
    forgets every earlier value and the recorded minimum. Each update takes O(1)
    time and memory.

    The paper uses ``>=`` in the warning and drift tests. If a context starts with
    more than ``min_num_items`` correct predictions, ``s_min`` is 0, and with ``>=``
    every later update would report a drift. The meter uses ``>``, as the river and
    MOA implementations do. With ``>``, the first error after such a run still
    reports a drift, as in the paper. The paper updates the minimum only when
    ``p_i + s_i`` is strictly lower, and the meter does the same. River also updates
    it on a tie, so the two can differ when two sums are exactly equal.

    ``remove`` is not supported, because the detector decides when it forgets values.

    Args:
        warning_level (float): The multiple of ``s_min`` in the warning test. The paper
            uses 2, a 95% confidence level (Default=2.0).
        drift_level (float): The multiple of ``s_min`` in the drift test. The paper
            uses 3, a 99% confidence level. It must be larger than ``warning_level``
            (Default=3.0).
        min_num_items (int): The number of values after which the tests start
            (Default=30).

    Raises:
        ValueError: If ``drift_level`` is not larger than ``warning_level``.

    Attributes:
        warning_level (float): The multiple of ``s_min`` in the warning test.
        drift_level (float): The multiple of ``s_min`` in the drift test.
        min_num_items (int): The number of values after which the tests start.
        num_items (int): The number of values since the last drift.
        drift_detected (bool): Whether the last update detected a drift.
        warning_detected (bool): Whether the last update detected a warning without a drift.
    """

    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_num_items: int = 30,
    ) -> None:
        if drift_level <= warning_level:
            raise ValueError("Expected warning_level < drift_level.")

        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_num_items = min_num_items
        self.drift_detected = False
        self.warning_detected = False
        self._reset()

    def _reset(self) -> None:
        self.num_items = 0
        self._num_errors = 0.0
        self._p_min = math.inf
        self._s_min = math.inf

    def update(self, num: float) -> DDMMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, 1 for an error and 0 for a correct prediction.

        Returns:
            object: self.
        """
        if self.drift_detected:
            self._reset()

        self.num_items += 1
        self._num_errors += num
        if self.num_items <= self.min_num_items:
            self.drift_detected = self.warning_detected = False
            return self

        p = self._num_errors / self.num_items
        s = math.sqrt(p * (1.0 - p) / self.num_items)
        if p + s < self._p_min + self._s_min:
            self._p_min, self._s_min = p, s

        self.drift_detected = p + s > self._p_min + self.drift_level * self._s_min
        self.warning_detected = (
            not self.drift_detected and p + s > self._p_min + self.warning_level * self._s_min
        )

        return self

    def remove(self, num: float) -> DDMMeter:
        """Raises ``NotImplementedError``, because the detector decides when it forgets values.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        raise NotImplementedError(
            "DDMMeter cannot remove observations: it forgets old values itself after a drift."
        )

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The error rate of the values since the last drift.
        """
        return self._num_errors / self.num_items
