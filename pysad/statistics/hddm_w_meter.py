from __future__ import annotations

import math

from pysad.core.base_statistic import UnivariateStatistic

# An exponentially weighted moving average and the sum of its squared weights, which is
# zero only before the first value.
_Sample = tuple[float, float]
_EMPTY: _Sample = (0.0, 0.0)


class HDDMWMeter(UnivariateStatistic):
    """The HDDM_W drift detector, which compares weighted moving averages with McDiarmid's bound.

    Follows HDDM with the W-test by Frías-Blanco et al. :cite:`friasblanco2015online`
    (Fig. 1 and Corollary 5). The meter keeps the exponentially weighted moving average
    (EWMA) ``Z`` of the values since the last drift, updated as
    ``Z = (1 - alpha) * Z + alpha * x`` from ``Z = x`` at the first value, and the sum
    ``D`` of its squared weights, updated as ``D = alpha**2 + (1 - alpha)**2 * D`` from
    ``D = 1``. The cut point is the value where ``Z + sqrt(D / 2 * ln(1 / drift_confidence))``
    was smallest. The meter keeps the EWMA ``X`` up to the cut point and the EWMA ``Y``
    of the values after it, each with its own ``D``. The W-test then reports a rise of
    the mean when::

        Y - X > sqrt((D_X + D_Y) / 2 * ln(1 / confidence))

    with ``confidence`` set to ``drift_confidence`` for a drift and to
    ``warning_confidence`` for a warning. With ``two_sided=True``, a second cut point
    at the largest lower bound ``Z - sqrt(D / 2 * ln(1 / drift_confidence))`` tests a
    fall of the mean in the same way. ``get`` returns ``Z``.

    The update after a drift starts a new run: it forgets every earlier value, as
    the counters reset in the paper. Each update takes O(1) time and memory.

    As in the paper, the bounds assume every value is in [0, 1]. Rescale values from
    a known range ``[a, b]`` to ``(x - a) / (b - a)`` before passing them to
    ``update``. The meter does not check the range.

    The paper does not give the confidence of the cut point bound. As in the authors'
    MOA implementation, the meter uses ``drift_confidence``. The river implementation
    sets ``D`` to ``alpha**2 + (1 - alpha)**2`` after the first value. The meter follows
    Section 5 of the paper and MOA with ``D = 1``. With this change, its drift and
    warning flags match river on the same stream. MOA and river move a cut point only
    when the bound is strictly smaller (or larger). The meter also moves it on a tie,
    as in Fig. 1.

    ``remove`` is not supported, because the detector decides when it forgets values.

    Args:
        drift_confidence (float): The significance level of the test for a drift,
            ``alpha_D`` in the paper (Default=0.001).
        warning_confidence (float): The significance level of the test for a warning,
            ``alpha_W`` in the paper. It must be larger than ``drift_confidence``
            (Default=0.005).
        alpha (float): The weight of each new value in the EWMA, ``lambda`` in the paper.
            It must be in the interval (0, 1] (Default=0.05, one of the two settings
            that the paper studies on real data).
        two_sided (bool): Whether to also detect a fall of the mean. The paper tests
            only for a rise (Default=False).

    Raises:
        ValueError: If not ``0 < drift_confidence < warning_confidence <= 1``, or if
            ``alpha`` is not in the interval (0, 1].

    Attributes:
        drift_confidence (float): The significance level of the test for a drift.
        warning_confidence (float): The significance level of the test for a warning.
        alpha (float): The weight of each new value in the EWMA.
        two_sided (bool): Whether a fall of the mean is also detected.
        drift_detected (bool): Whether the last update detected a drift.
        warning_detected (bool): Whether the last update detected a warning without a drift.
    """

    def __init__(
        self,
        drift_confidence: float = 0.001,
        warning_confidence: float = 0.005,
        alpha: float = 0.05,
        two_sided: bool = False,
    ) -> None:
        if not 0.0 < drift_confidence < warning_confidence <= 1.0:
            raise ValueError("Expected 0 < drift_confidence < warning_confidence <= 1.")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the interval (0, 1].")

        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self.alpha = alpha
        self.two_sided = two_sided
        self.drift_detected = False
        self.warning_detected = False
        self._reset()

    def _reset(self) -> None:
        self._total = _EMPTY
        # The lowest upper bound and the largest lower bound of the EWMA so far, and the
        # EWMAs up to and after the cut points where they were reached.
        self._low_bound, self._low_before, self._low_after = math.inf, _EMPTY, _EMPTY
        self._high_bound, self._high_before, self._high_after = -math.inf, _EMPTY, _EMPTY

    def _add(self, sample: _Sample, num: float) -> _Sample:
        ewma, d = sample
        if d == 0.0:
            return num, 1.0

        keep = 1.0 - self.alpha
        return keep * ewma + self.alpha * num, self.alpha * self.alpha + keep * keep * d

    def update(self, num: float) -> HDDMWMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        if self.drift_detected:
            self._reset()

        self._total = ewma, d = self._add(self._total, num)
        eps = math.sqrt(d / 2.0 * math.log(1.0 / self.drift_confidence))

        if ewma + eps <= self._low_bound:
            self._low_bound, self._low_before, self._low_after = ewma + eps, self._total, _EMPTY
        else:
            self._low_after = self._add(self._low_after, num)

        if ewma - eps >= self._high_bound:
            self._high_bound, self._high_before, self._high_after = ewma - eps, self._total, _EMPTY
        else:
            self._high_after = self._add(self._high_after, num)

        self.drift_detected = self._changed(self.drift_confidence)
        self.warning_detected = not self.drift_detected and self._changed(self.warning_confidence)

        return self

    def _changed(self, confidence: float) -> bool:
        if _rises(self._low_before, self._low_after, confidence):
            return True

        return self.two_sided and _rises(self._high_after, self._high_before, confidence)

    def remove(self, num: float) -> HDDMWMeter:
        """Raises ``NotImplementedError``, because the detector decides when it forgets values.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        raise NotImplementedError(
            "HDDMWMeter cannot remove observations: it forgets old values itself after a drift."
        )

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The EWMA of the values since the last drift.
        """
        return self._total[0]


def _rises(before: _Sample, after: _Sample, confidence: float) -> bool:
    # The W-test (Corollary 5) for E[after] > E[before].
    if before[1] == 0.0 or after[1] == 0.0:
        return False

    eps = math.sqrt((before[1] + after[1]) / 2.0 * math.log(1.0 / confidence))
    return after[0] - before[0] > eps
