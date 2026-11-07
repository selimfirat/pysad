from __future__ import annotations

import math

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state

from pysad.core.base_statistic import UnivariateStatistic


class KSWINMeter(UnivariateStatistic):
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
        ValueError: If ``alpha`` is not in (0, 1).

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

    def remove(self, num: float) -> KSWINMeter:
        """Raises ``NotImplementedError``, because KSWIN decides which values leave its window.

        Args:
            num (float): The value to be removed.

        Returns:
            object: self.
        """
        raise NotImplementedError(
            "KSWINMeter cannot remove observations: it drops old values itself."
        )

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The Kolmogorov-Smirnov distance of the last test, or 0.0 before the window is full.
        """
        return self._dist
