from __future__ import annotations

import math

from pysad.core.base_statistic import UnivariateStatistic


class HDDMAMeter(UnivariateStatistic):
    """The HDDM_A drift detector, which compares moving averages with Hoeffding's bound.

    Follows HDDM with the A-test by Frías-Blanco et al. :cite:`friasblanco2015online`
    (Fig. 1 and Corollary 9). The meter keeps the mean ``Z`` of the ``n`` values since
    the last drift and a cut point: the prefix of ``cut`` values whose mean ``X`` had
    the smallest upper bound ``X + sqrt(ln(1 / drift_confidence) / (2 * cut))``. Each
    update moves the cut point to the current value when ``Z`` reaches a smaller or
    equal upper bound. The A-test then reports a rise of the mean when::

        Z - X >= sqrt((n - cut) / (2 * cut * n) * ln(1 / confidence))

    with ``confidence`` set to ``drift_confidence`` for a drift and to
    ``warning_confidence`` for a warning. With ``two_sided=True``, a second cut point
    at the largest lower bound tests a fall of the mean in the same way, as the paper
    describes after Corollary 3. ``get`` returns ``Z``.

    The update after a drift starts a new run: it forgets every earlier value, as
    the counters reset in the paper. Each update takes O(1) time and memory.

    As in the paper, the bounds assume every value is in [0, 1]. Rescale values from
    a known range ``[a, b]`` to ``(x - a) / (b - a)`` before passing them to
    ``update``. The meter does not check the range.

    The paper does not give the confidence of the cut point bound. As in the authors'
    MOA implementation, the meter uses ``drift_confidence``. The MOA and river
    implementations use ``ln(2 / confidence)`` in the A-test. The meter follows
    Corollary 9 of the paper and uses ``ln(1 / confidence)``, so it detects a drift a
    little earlier. With this change, its drift and warning flags match river on the
    same stream. With ``two_sided=True``, MOA resets after a fall of the mean but does
    not report a drift. The meter reports it, as river does.

    ``remove`` is not supported, because the detector decides when it forgets values.

    Args:
        drift_confidence (float): The significance level of the test for a drift,
            ``alpha_D`` in the paper (Default=0.001).
        warning_confidence (float): The significance level of the test for a warning,
            ``alpha_W`` in the paper. It must be larger than ``drift_confidence``
            (Default=0.005).
        two_sided (bool): Whether to also detect a fall of the mean. The paper tests
            only for a rise (Default=False).

    Raises:
        ValueError: If not ``0 < drift_confidence < warning_confidence <= 1``.

    Attributes:
        drift_confidence (float): The significance level of the test for a drift.
        warning_confidence (float): The significance level of the test for a warning.
        two_sided (bool): Whether a fall of the mean is also detected.
        num_items (int): The number of values since the last drift.
        drift_detected (bool): Whether the last update detected a drift.
        warning_detected (bool): Whether the last update detected a warning without a drift.
    """

    def __init__(
        self,
        drift_confidence: float = 0.001,
        warning_confidence: float = 0.005,
        two_sided: bool = False,
    ) -> None:
        if not 0.0 < drift_confidence < warning_confidence <= 1.0:
            raise ValueError("Expected 0 < drift_confidence < warning_confidence <= 1.")

        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.two_sided = two_sided
        self.drift_detected = False
        self.warning_detected = False
        self._reset()

    def _reset(self) -> None:
        self.num_items = 0
        self._sum = 0.0
        # The number of values and their sum up to the cut points for a rise and a fall.
        self._low_n, self._low_sum = 0, 0.0
        self._high_n, self._high_sum = 0, 0.0

    def update(self, num: float) -> HDDMAMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        if self.drift_detected:
            self._reset()

        self.num_items += 1
        self._sum += num
        n, mean = self.num_items, self._sum / self.num_items

        log_drift = math.log(1.0 / self.drift_confidence)
        eps = math.sqrt(log_drift / (2 * n))
        if self._low_n == 0 or (
            self._low_sum / self._low_n + math.sqrt(log_drift / (2 * self._low_n)) >= mean + eps
        ):
            self._low_n, self._low_sum = n, self._sum
        if self._high_n == 0 or (
            self._high_sum / self._high_n - math.sqrt(log_drift / (2 * self._high_n)) <= mean - eps
        ):
            self._high_n, self._high_sum = n, self._sum

        self.drift_detected = self._changed(self.drift_confidence)
        self.warning_detected = not self.drift_detected and self._changed(self.warning_confidence)

        return self

    def _changed(self, confidence: float) -> bool:
        n, mean = self.num_items, self._sum / self.num_items
        log_term = math.log(1.0 / confidence)
        if self._low_n < n:
            eps = math.sqrt((n - self._low_n) / (2 * self._low_n * n) * log_term)
            if mean - self._low_sum / self._low_n >= eps:
                return True

        if self.two_sided and self._high_n < n:
            eps = math.sqrt((n - self._high_n) / (2 * self._high_n * n) * log_term)
            return self._high_sum / self._high_n - mean >= eps

        return False

    def remove(self, num: float) -> HDDMAMeter:
        """Raises ``NotImplementedError``, because the detector decides when it forgets values.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        raise NotImplementedError(
            "HDDMAMeter cannot remove observations: it forgets old values itself after a drift."
        )

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The mean of the values since the last drift.
        """
        return self._sum / self.num_items
