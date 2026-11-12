from __future__ import annotations

import math
from collections import deque

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state

from pysad.core.base_statistic import BaseStatistic

# An exponentially weighted moving average and the sum of its squared weights, which is
# zero only before the first value (used by HDDMWMeter).
_Sample = tuple[float, float]
_EMPTY: _Sample = (0.0, 0.0)


def _rises(before: _Sample, after: _Sample, confidence: float) -> bool:
    # The W-test (Corollary 5) for E[after] > E[before].
    if before[1] == 0.0 or after[1] == 0.0:
        return False

    eps = math.sqrt((before[1] + after[1]) / 2.0 * math.log(1.0 / confidence))
    return after[0] - before[0] > eps


class ADWINMeter(BaseStatistic):
    """The mean of an adaptive window that shrinks when the stream changes.

    Follows ADWIN2 by Bifet and Gavaldà :cite:`bifet2007learning`. The window
    ``W`` holds the most recent items and grows while the stream is stationary.
    After each update, every split of ``W`` into an older part ``W0`` and a newer
    part ``W1`` at a bucket border is tested, and the oldest bucket is dropped
    while some split has ``|mean(W0) - mean(W1)| > eps_cut``, with the paper's
    normal-approximation bound (Eq. 3.1)::

        eps_cut = sqrt(2 / m * var(W) * ln(2 / delta')) + 2 / (3 * m) * ln(2 / delta')

    where ``1 / m = 1 / n0 + 1 / n1`` for the split lengths ``n0`` and ``n1``,
    ``var(W)`` is the variance of the window, and ``delta' = delta / ln(n)`` for
    the window length ``n``. ``get`` returns the mean of the window.

    As in the paper, the bound assumes every value is in [0, 1]: its second term
    does not scale with the data. Rescale values from a known range ``[a, b]`` to
    ``(x - a) / (b - a)`` before passing them to ``update``. The meter does not
    check the range.

    The window is kept as an exponential histogram: buckets whose sizes are powers
    of two, with at most ``max_buckets`` buckets of each size. When a size has
    ``max_buckets + 1`` buckets, its two oldest are merged into one bucket of the
    next size, so memory and the number of tested splits are O(log W).

    Deviation from the paper: each bucket also keeps the sum of squared deviations
    of its items, so ``var(W)`` is exact after buckets are merged or dropped. The
    paper stores only the size and the sum of each bucket and does not say how
    ``var(W)`` is maintained.

    ``remove`` is not supported, because ADWIN decides which items leave its window.

    Args:
        delta (float): The confidence of the change test, in (0, 1). Smaller values
            give fewer false alarms and slower detection (Default=0.002, the default
            of MOA and river; the paper's experiments use 0.05, 0.1 and 0.3).
        max_buckets (int): The largest number of buckets of each size, ``M`` in the
            paper (Default=5).

    Raises:
        ValueError: If ``delta`` is not in (0, 1).

    Attributes:
        delta (float): The confidence of the change test.
        max_buckets (int): The largest number of buckets of each size.
        width (int): The number of items in the window.
        drift_detected (bool): Whether the last update shrank the window.
    """

    def __init__(self, delta: float = 0.002, max_buckets: int = 5) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1).")

        self.delta = delta
        self.max_buckets = max_buckets
        self.width = 0
        self.drift_detected = False
        self._sum = 0.0
        self._sq_dev = 0.0  # Sum of squared deviations from the window mean.
        # _buckets[i] holds the [sum, sum of squared deviations] of each bucket of 2**i items, oldest first.
        self._buckets: list[list[list[float]]] = []

    def update(self, num: float) -> ADWINMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        if self.width > 0:
            diff = num - self._sum / self.width
            self._sq_dev += diff * diff * self.width / (self.width + 1)
        self._sum += num
        self.width += 1

        if not self._buckets:
            self._buckets.append([])
        self._buckets[0].append([num, 0.0])
        self._compress()

        self.drift_detected = False
        while self._has_cut():
            self._drop_oldest()
            self.drift_detected = True

        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The mean of the window.
        """
        return self._sum / self.width

    def _compress(self) -> None:
        for level, buckets in enumerate(self._buckets):
            if len(buckets) <= self.max_buckets:
                return

            size = 2**level
            (sum0, sq_dev0), (sum1, sq_dev1) = buckets.pop(0), buckets.pop(0)
            merged = [sum0 + sum1, sq_dev0 + sq_dev1 + (sum0 - sum1) ** 2 / (2 * size)]
            if level + 1 == len(self._buckets):
                self._buckets.append([])
            self._buckets[level + 1].append(merged)

    def _has_cut(self) -> bool:
        n = self.width
        if n < 2:
            return False

        variance = self._sq_dev / n
        log_term = math.log(2.0 * math.log(n) / self.delta)
        n0, sum0 = 0, 0.0
        for level in range(len(self._buckets) - 1, -1, -1):
            for bucket_sum, _ in self._buckets[level]:
                n0 += 2**level
                sum0 += bucket_sum
                n1 = n - n0
                if n1 == 0:
                    return False

                inv_m = 1.0 / n0 + 1.0 / n1
                eps_cut = (
                    math.sqrt(2.0 * inv_m * variance * log_term) + 2.0 / 3.0 * inv_m * log_term
                )
                if abs(sum0 / n0 - (self._sum - sum0) / n1) > eps_cut:
                    return True

        return False

    def _drop_oldest(self) -> None:
        size = 2 ** (len(self._buckets) - 1)
        bucket_sum, bucket_sq_dev = self._buckets[-1].pop(0)
        rest = self.width - size
        diff = bucket_sum / size - (self._sum - bucket_sum) / rest
        self._sq_dev = max(
            0.0, self._sq_dev - bucket_sq_dev - diff * diff * size * rest / self.width
        )
        self._sum -= bucket_sum
        self.width = rest
        while not self._buckets[-1]:
            self._buckets.pop()


class DDMMeter(BaseStatistic):
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

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The error rate of the values since the last drift.
        """
        return self._num_errors / self.num_items


class EDDMMeter(BaseStatistic):
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

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The mean distance between two errors since the last drift, or 0.0
            before the first error.
        """
        return self._mean


class FHDDMMeter(BaseStatistic):
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
        ValueError: If ``window_size`` is less than 1 or ``delta`` is not in (0, 1).

    Attributes:
        window_size (int): The number of the last values in the window.
        delta (float): The probability bound of the Hoeffding test.
        epsilon (float): The Hoeffding bound ``epsilon_d`` of the drift test.
        num_items (int): The number of values since the last drift.
        drift_detected (bool): Whether the last update detected a drift.
    """

    def __init__(self, window_size: int = 100, delta: float = 1e-6) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")
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

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The error rate of the values in the window.
        """
        return self._num_errors / len(self._window)


class HDDMAMeter(BaseStatistic):
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

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The mean of the values since the last drift.
        """
        return self._sum / self.num_items


class HDDMWMeter(BaseStatistic):
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

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The EWMA of the values since the last drift.
        """
        return self._total[0]


class KSWINMeter(BaseStatistic):
    """The Kolmogorov-Smirnov distance between the recent and the older values of a sliding window.

    Follows KSWIN (Kolmogorov-Smirnov Windowing) by Raab et al. :cite:`raab2020reactive`.
    A sliding window ``Psi`` keeps the ``window_size`` most recent values. When the
    window is full, each update takes its ``stat_size`` newest values as ``R`` and
    draws ``stat_size`` values ``W`` uniformly, without replacement, from the older
    ``window_size - stat_size`` values (Eq. 3). It then computes the two-sample
    Kolmogorov-Smirnov distance (Eq. 4)::

        dist = sup_x |F_W(x) - F_R(x)|

    and detects a drift if ``dist > sqrt(-ln(alpha) / stat_size)`` (Eq. 5). ``get``
    returns the distance of the last test, or 0.0 before the window is full. The
    test compares whole distributions, so it also detects a change in the spread
    or the shape of the stream when the mean stays the same.

    The paper does not say what happens to the window after a drift. As in the
    authors' scikit-multiflow implementation, the window keeps only ``R``, the
    values of the new concept, and the test runs again when the window is full.

    Deviation from the scikit-multiflow and river implementations: they detect a
    drift if the p-value of SciPy's ``ks_2samp`` is at most ``alpha`` and the
    distance is above 0.1. This meter uses the threshold of Eq. 5 of the paper.

    The meter is univariate. The paper runs one test for each feature and detects
    a drift if any feature drifts. Use one meter for each feature to do the same.

    ``remove`` is not supported, because KSWIN decides which values leave its window.

    Args:
        alpha (float): The significance level of the test, in (0, 1). Smaller values
            give fewer false alarms and slower detection (Default=0.0001, the value
            of the paper's experiments, after a Bonferroni correction; river uses 0.005).
        window_size (int): The number of values in the sliding window, ``n`` in the
            paper, at least ``2 * stat_size`` (Default=300, as in the paper; river uses 100).
        stat_size (int): The number of values in each tested sample, ``r`` in the
            paper (Default=30, as in the paper and river).
        random_state (int, RandomState or None): The seed or the random state
            that draws ``W``. ``None`` uses NumPy's global random state (Default=None).

    Raises:
        ValueError: If ``alpha`` is not in (0, 1), or ``window_size`` is less than
            ``2 * stat_size``.

    Attributes:
        alpha (float): The significance level of the test.
        window_size (int): The number of values in the sliding window.
        stat_size (int): The number of values in each tested sample.
        random_state (int, RandomState or None): The seed or the random state that draws ``W``.
        drift_detected (bool): Whether the last update detected a drift.
    """

    def __init__(
        self,
        alpha: float = 0.0001,
        window_size: int = 300,
        stat_size: int = 30,
        random_state: int | RandomState | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        # W is drawn without replacement from the window_size - stat_size older values.
        if window_size < 2 * stat_size:
            raise ValueError("window_size must be at least 2 * stat_size.")

        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.random_state = random_state
        self.drift_detected = False
        self._rng = check_random_state(random_state)
        self._threshold = math.sqrt(-math.log(alpha) / stat_size)
        self._window = np.empty(window_size)  # Oldest value first.
        self._size = 0
        self._dist = 0.0

    def update(self, num: float) -> KSWINMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        n, r = self.window_size, self.stat_size
        self.drift_detected = False
        if self._size < n:
            self._window[self._size] = num
            self._size += 1
            if self._size < n:
                return self
        else:
            self._window[:-1] = self._window[1:]
            self._window[-1] = num

        older = self._window[self._rng.choice(n - r, size=r, replace=False)]
        recent = self._window[n - r :]
        older.sort()
        recent = np.sort(recent)
        # Both samples have r values, so the distance is the largest gap in counts over all points.
        points = np.concatenate((older, recent))
        gaps = older.searchsorted(points, side="right") - recent.searchsorted(points, side="right")
        self._dist = float(np.abs(gaps).max()) / r

        if self._dist > self._threshold:
            self.drift_detected = True
            self._window[:r] = self._window[n - r :]
            self._size = r

        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The Kolmogorov-Smirnov distance of the last test, or 0.0 before the window is full.
        """
        return self._dist


class PageHinkleyMeter(BaseStatistic):
    """The Page-Hinkley statistic for sequential change-point detection.

    Tracks the cumulative deviation of the stream from its running mean in
    O(1) time and memory per observation, following Page :cite:`page1954continuous`.
    Two one-sided sums are kept, one for upward shifts and one for downward shifts::

        g+_t = max(0, g+_{t-1} + x_t - mean_t - delta)
        g-_t = min(0, g-_{t-1} + x_t - mean_t + delta)

    where ``mean_t`` is the running mean including the new observation and
    ``delta`` is the smallest shift magnitude worth accumulating. By Lindley's
    identity, clamping at zero is algebraically equivalent to Page's formulation
    of subtracting the running minimum (or maximum) of the unreflected cumulative
    sum, without storing the running extrema separately. ``get`` returns
    ``max(g+, -g-)``, the two-sided statistic: a caller raises a change-point
    alarm when it exceeds a chosen threshold. Small fluctuations below ``delta``
    decay back toward zero instead of accumulating.

    ``remove`` is not supported: the running mean and the cumulative sums depend
    on the full history, so a past observation cannot be taken back.

    Args:
        delta (float): The smallest shift magnitude to accumulate, which must be
            non-negative (Default=0.0, every deviation accumulates).

    Raises:
        ValueError: If ``delta`` is negative.

    Attributes:
        delta (float): The smallest shift magnitude that is accumulated.
        num_items (int): The number of items that are used to update the statistic.
        mean (float): The running mean of the observations seen so far.
        g_plus (float): The upward cumulative sum (always non-negative).
        g_minus (float): The downward cumulative sum (always non-positive).
    """

    def __init__(self, delta: float = 0.0) -> None:
        if delta < 0.0:
            raise ValueError("delta must be non-negative.")

        self.delta = delta
        self.num_items = 0
        self.mean = 0.0
        self.g_plus = 0.0
        self.g_minus = 0.0

    def update(self, num: float) -> PageHinkleyMeter:
        """Updates the statistic with the value for a timestep.

        Args:
            num (float): The incoming value, for which the statistic is used.

        Returns:
            object: self.
        """
        self.num_items += 1
        self.mean += (num - self.mean) / self.num_items

        deviation = num - self.mean
        self.g_plus = max(0.0, self.g_plus + deviation - self.delta)
        self.g_minus = min(0.0, self.g_minus + deviation + self.delta)

        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The two-sided Page-Hinkley statistic, ``max(g+, -g-)``.
        """
        return max(self.g_plus, -self.g_minus)
