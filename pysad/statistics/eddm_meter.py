from __future__ import annotations

import math

from pysad.core.base_statistic import UnivariateStatistic


class EDDMMeter(UnivariateStatistic):
    """The EDDM drift detector, which tracks the distance between two errors.

    Follows the Early Drift Detection Method by Baena-García et al. :cite:`baenagarcia2006early`
    (Section 3.2). The meter reads a stream of bits, in which a nonzero value is an error,
    for example a wrong prediction of a classifier. At each error, it updates the mean
    ``p`` and the standard deviation ``s`` of the distances between consecutive errors,
    and stores the largest value ``p_max + 2 * s_max`` of ``p + 2 * s``. A model that
    learns makes errors less often, so the distances grow. When the errors come more
    often again, the ratio::

        (p + 2 * s) / (p_max + 2 * s_max)

    falls. The meter reports a warning while the ratio is below ``warning_threshold``,
    and a drift when it is below ``drift_threshold``. As in the paper, the warning stays
    on until an error brings the ratio back over ``warning_threshold``. ``get`` returns
    ``p``.

    The paper searches for a drift only after ``min_num_errors`` errors, because it
    first estimates the distribution of the distances. The meter therefore starts to
    store the maximum and to test at the error number ``min_num_errors``. The paper
    leaves the other details open. The meter follows the MOA implementation of the
    authors for them: the distance is the difference of the positions of two errors, so
    two consecutive errors have distance 1, the first distance counts from the start of
    the run, and ``s`` is the population standard deviation.

    The river implementation differs in four points. It stores the maximum from the
    31st value on, and not from the error number ``min_num_errors``. It tests from the
    error number ``min_num_errors + 1``. It uses the sample standard deviation. It does
    not change the warning flag at an error that sets a new maximum or detects a drift.
    With these four changes, the drift and warning flags of the meter match river on
    the same stream. The meter has fewer false drifts than river on streams without a
    change, because it does not store a maximum from the first, noisy estimates.

    The update after a drift starts a new run: it forgets every earlier value, as the
    paper resets ``p_max`` and ``s_max``. Each update takes O(1) time and memory.

    ``remove`` is not supported, because the detector decides when it forgets values.

    Args:
        warning_threshold (float): The ratio below which a warning is reported, ``alpha``
            in the paper (Default=0.95).
        drift_threshold (float): The ratio below which a drift is reported, ``beta`` in
            the paper. It must not be larger than ``warning_threshold`` (Default=0.9).
        min_num_errors (int): The number of errors before the first test (Default=30).

    Raises:
        ValueError: If not ``0 < drift_threshold <= warning_threshold <= 1``.

    Attributes:
        warning_threshold (float): The ratio below which a warning is reported.
        drift_threshold (float): The ratio below which a drift is reported.
        min_num_errors (int): The number of errors before the first test.
        num_items (int): The number of values since the last drift.
        num_errors (int): The number of errors since the last drift.
        drift_detected (bool): Whether the last update detected a drift.
        warning_detected (bool): Whether the meter is in the warning level and the last
            update did not detect a drift.
    """

    def __init__(
        self,
        warning_threshold: float = 0.95,
        drift_threshold: float = 0.9,
        min_num_errors: int = 30,
    ) -> None:
        if not 0.0 < drift_threshold <= warning_threshold <= 1.0:
            raise ValueError("Expected 0 < drift_threshold <= warning_threshold <= 1.")

        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        self.min_num_errors = min_num_errors
        self.drift_detected = False
        self._reset()

    def _reset(self) -> None:
        self.num_items = 0
        self.num_errors = 0
        self.warning_detected = False
        self._last_error = 0
        # The running mean and sum of squared deviations of the distances (Welford).
        self._mean = 0.0
        self._sum_squares = 0.0
        self._max_level = 0.0

    def update(self, num: float) -> EDDMMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        if self.drift_detected:
            self.drift_detected = False
            self._reset()

        self.num_items += 1
        if not num:
            return self

        self.num_errors += 1
        distance = self.num_items - self._last_error
        self._last_error = self.num_items
        delta = distance - self._mean
        self._mean += delta / self.num_errors
        self._sum_squares += delta * (distance - self._mean)
        if self.num_errors < self.min_num_errors:
            return self

        level = self._mean + 2.0 * math.sqrt(self._sum_squares / self.num_errors)
        self._max_level = max(self._max_level, level)
        ratio = level / self._max_level
        self.drift_detected = ratio < self.drift_threshold
        self.warning_detected = not self.drift_detected and ratio < self.warning_threshold

        return self

    def remove(self, num: float) -> EDDMMeter:
        """Raises ``NotImplementedError``, because the detector decides when it forgets values.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        raise NotImplementedError(
            "EDDMMeter cannot remove observations: it forgets old values itself after a drift."
        )

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The mean distance between two errors since the last drift, or 0.0
            before the first error.
        """
        return self._mean
