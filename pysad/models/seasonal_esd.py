from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.stats import t

from pysad.core.base_model import BaseModel
from pysad.transform.preprocessing import ModifiedSTLResidualTransformer
from pysad.utils.validation import check_univariate
from pysad.utils.window import RingBuffer, _median_and_mad_sorted


@lru_cache(maxsize=1024)
def _critical_value(alpha: float, num_tails: float, n: int, i: int) -> float:
    # The critical value of the i-th ESD test on n values. It depends only on these numbers, so it
    # is computed once instead of once per scored instance.
    p = 1.0 - alpha / (num_tails * (n - i + 1))
    t_value = t.ppf(p, n - i - 1)
    return ((n - i) * t_value) / np.sqrt((n - i - 1 + t_value**2) * (n - i + 1))


def _median_mad_sorted(sub: np.ndarray) -> tuple[float, float]:
    """Returns ``(median, 1.4826 * MAD)`` of sorted 1-D slice ``sub`` in O(log m) time."""
    center, mad = _median_and_mad_sorted(sub)
    return center, 1.4826 * mad


class SeasonalESD(BaseModel):
    """Flags seasonal anomalies in a univariate stream with the Seasonal ESD (S-ESD) test over a sliding window :cite:`hochenbaum2017automatic`.

    This is the S-ESD method of the paper (Eq. 5). It computes the modified STL residual ``X - seasonal - median(X)`` over a fixed-size window, with the periodic STL of R (see :class:`pysad.transform.preprocessing.ModifiedSTLResidualTransformer`), then applies generalized ESD with the mean and the standard deviation to the residuals. The reference implementation of the paper, Twitter's AnomalyDetection R package, has no S-ESD: it always uses the median and the MAD, as :class:`SeasonalHybridESD` does. This model follows the R package in everything else. The test stops when the scale of the remaining values is at most ``window_size`` times the float64 machine epsilon times the largest absolute value in the window, a bound on the rounding error of the residuals, where R stops only when the MAD is exactly 0, so that the floating-point noise of a series without anomalies raises no alerts. The paper runs the test on a full batch. This model runs it on the most recent ``window_size`` values for each instance.

    The score of an instance is the ESD test statistic of the newest value if the test flags it as an anomaly, else 0.0. The score is 0.0 until the window holds ``window_size`` values.

    Args:
        period (int): Number of observations in one seasonal period. It must be at least 2.
        window_size (int): Number of recent observations used for STL and ESD. It must be greater than ``2 * period``.
        max_anomalies (float): Maximum number of anomalies that ESD tests in each window, as a fraction of ``window_size``, greater than 0 and at most 0.49. The test looks for ``max(1, int(max_anomalies * window_size))`` anomalies (Default=0.1).
        alpha (float): ESD significance level, strictly between 0 and 1 (Default=0.05).
        direction (str): Which anomalies the test flags: ``"pos"`` for high values, ``"neg"`` for low values, or ``"both"`` for both, with the critical values of a two-sided test (Default="pos").
        robust (bool): Whether to use robust STL fitting (Default=True).

    Raises:
        ValueError: If ``period`` is less than 2, ``window_size`` is at most ``2 * period``, ``max_anomalies`` is not greater than 0 and at most 0.49, ``alpha`` is not strictly between 0 and 1, or ``direction`` is not ``"pos"``, ``"neg"`` or ``"both"``.
    """

    _use_median = False

    def __init__(
        self,
        period: int,
        window_size: int,
        max_anomalies: float = 0.1,
        alpha: float = 0.05,
        direction: str = "pos",
        robust: bool = True,
    ) -> None:
        if window_size <= 2 * period:
            raise ValueError("window_size must be greater than 2 * period.")

        if not 0 < max_anomalies <= 0.49:
            raise ValueError("max_anomalies must be greater than 0 and at most 0.49.")

        if not np.isfinite(alpha) or not (0 < alpha < 1):
            raise ValueError("alpha must be finite and between 0 and 1.")

        if direction not in ("pos", "neg", "both"):
            raise ValueError("direction must be 'pos', 'neg' or 'both'.")

        self.period = period
        self.window_size = window_size
        self.max_anomalies = max_anomalies
        self.alpha = alpha
        self.direction = direction
        self._window = RingBuffer(window_size)
        self.residual_transformer = ModifiedSTLResidualTransformer(
            period=period, window_size=window_size, robust=robust
        )

    def _as_value(self, X: np.ndarray) -> float:
        return check_univariate(X, "SeasonalESD")

    def _center_scale(self, values: np.ndarray) -> tuple[float, float]:
        return float(values.mean()), float(values.std(ddof=1))

    def _critical_value(self, n: int, i: int) -> float:
        return _critical_value(self.alpha, 2.0 if self.direction == "both" else 1.0, n, i)

    def _esd_target(self, values: np.ndarray, min_scale: float, target_idx: int) -> float:
        arr = np.asarray(values, dtype=np.float64)
        n = arr.shape[0]
        max_k = max(1, int(self.max_anomalies * n))
        orig_idx = np.argsort(arr, kind="stable")
        s = arr[orig_idx]

        use_median = self._use_median
        mid = n >> 1
        ref = 0.0
        left_s1 = right_s1 = left_s2 = right_s2 = arr
        if not use_median:
            ref = float(s[mid])
            d = s - ref
            right_len = n - mid
            right_s1 = np.empty(right_len + 1, dtype=np.float64)
            right_s2 = np.empty(right_len + 1, dtype=np.float64)
            right_s1[0] = 0.0
            right_s2[0] = 0.0
            dr = d[mid:]
            np.cumsum(dr, out=right_s1[1:])
            np.cumsum(dr * dr, out=right_s2[1:])

            left_s1 = np.empty(mid + 1, dtype=np.float64)
            left_s2 = np.empty(mid + 1, dtype=np.float64)
            left_s1[mid] = 0.0
            left_s2[mid] = 0.0
            if mid > 0:
                dl_rev = d[mid - 1 :: -1]
                left_s1[mid - 1 :: -1] = np.cumsum(dl_rev)
                left_s2[mid - 1 :: -1] = np.cumsum(dl_rev * dl_rev)

        lo = 0
        hi = n - 1
        direction = self.direction
        target_stat: float | None = None

        for i in range(1, max_k + 1):
            m = hi - lo + 1
            if m <= 1:
                break
            if use_median:
                center, scale = _median_mad_sorted(s[lo : hi + 1])
            elif lo <= mid <= hi + 1:
                sum1 = float(left_s1[lo] + right_s1[hi + 1 - mid])
                sum2 = float(left_s2[lo] + right_s2[hi + 1 - mid])
                mean_shifted = sum1 / m
                center = ref + mean_shifted
                var = (sum2 - sum1 * mean_shifted) / (m - 1)
                scale = float(np.sqrt(var)) if var > 0.0 else 0.0
            else:
                center, scale = self._center_scale(s[lo : hi + 1])

            if scale <= min_scale:
                break

            if direction == "pos":
                pick_low = False
                statistic = (float(s[hi]) - center) / scale
            elif direction == "neg":
                pick_low = True
                statistic = (center - float(s[lo])) / scale
            else:
                stat_lo = abs(float(s[lo]) - center) / scale
                stat_hi = abs(float(s[hi]) - center) / scale
                if stat_lo > stat_hi:
                    pick_low = True
                    statistic = stat_lo
                elif stat_hi > stat_lo:
                    pick_low = False
                    statistic = stat_hi
                else:
                    min_lo = int(orig_idx[lo])
                    if s[hi - 1] == s[hi]:
                        p_tie = lo + int(np.searchsorted(s[lo : hi + 1], s[hi], side="left"))
                        min_hi = int(orig_idx[p_tie])
                    else:
                        min_hi = int(orig_idx[hi])
                    pick_low = min_lo < min_hi
                    statistic = stat_lo

            if pick_low:
                peeled_idx = int(orig_idx[lo])
                lo += 1
            elif hi > lo and s[hi - 1] == s[hi]:
                p = lo + int(np.searchsorted(s[lo : hi + 1], s[hi], side="left"))
                peeled_idx = int(orig_idx[p])
                orig_idx[p:hi] = orig_idx[p + 1 : hi + 1]
                hi -= 1
            else:
                peeled_idx = int(orig_idx[hi])
                hi -= 1

            critical_value = self._critical_value(n, i)
            if peeled_idx == target_idx:
                if statistic > critical_value:
                    return float(statistic)
                target_stat = float(statistic)
            elif target_stat is not None and statistic > critical_value:
                return target_stat

        return 0.0

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> SeasonalESD:
        """Adds the next instance to the model window.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.
            y (int or None): Ignored. Present for API consistency (Default=None).

        Returns:
            object: self.
        """
        self._window.append(self._as_value(X))
        return self

    def _score_window(self, values: np.ndarray) -> float:
        residuals = self.residual_transformer.transform_window(values)
        # The rounding error of the residuals grows with the window size and the largest value.
        return self._esd_target(
            residuals,
            min_scale=self.window_size * np.finfo(np.float64).eps * float(np.abs(values).max()),
            target_idx=values.shape[0] - 1,
        )

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the next instance in the window that ends with it, without adding it to the window.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.

        Returns:
            float: The ESD test statistic of ``X`` if the test flags it as an anomaly, else 0.0.
        """
        value = self._as_value(X)
        if len(self._window) + 1 < self.window_size:
            return 0.0

        first, second, num_prior = self._window.active_slices()
        candidate = np.empty(num_prior + 1, dtype=np.float64)
        n_first = first.shape[0]
        candidate[:n_first] = first
        if second is not None:
            candidate[n_first:num_prior] = second
        candidate[num_prior] = value
        return self._score_window(candidate)

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Adds the next instance to the window and scores it.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.
            y (int or None): Ignored. Present for API consistency (Default=None).

        Returns:
            float: The ESD test statistic of ``X`` if the test flags it as an anomaly, else 0.0.
        """
        self.fit_partial(X, y)
        if len(self._window) < self.window_size:
            return 0.0
        return self._score_window(self._window.to_array())


class SeasonalHybridESD(SeasonalESD):
    """Flags seasonal anomalies in a univariate stream with the Seasonal Hybrid ESD (S-H-ESD) test over a sliding window :cite:`hochenbaum2017automatic`.

    This is the S-H-ESD method of the paper. It uses the same periodic modified STL residual as :class:`SeasonalESD`, then replaces the mean and the standard deviation of ESD with the median and the MAD-based scale (1.4826 times the MAD) in the test statistic. For each window, the test flags the anomalies that ``AnomalyDetectionVec(window, max_anoms = max_anomalies, direction = direction, alpha = alpha, period = period)`` of Twitter's AnomalyDetection R package, the reference implementation of the paper, flags. The exception is an even ``window_size``, for which the robust STL sometimes differs from R, as :class:`pysad.transform.preprocessing.ModifiedSTLResidualTransformer` documents. Use an odd ``window_size`` to match R. The test stops when the scale of the remaining values is at most ``window_size`` times the float64 machine epsilon times the largest absolute value in the window, a bound on the rounding error of the residuals, where R stops only when the MAD is exactly 0, so that the floating-point noise of a series without anomalies raises no alerts. The paper runs the test on a full batch. This model runs it on the most recent ``window_size`` values for each instance.

    The score of an instance is the test statistic of the newest value if the test flags it as an anomaly, else 0.0. The score is 0.0 until the window holds ``window_size`` values.

    Args:
        period (int): Number of observations in one seasonal period. It must be at least 2.
        window_size (int): Number of recent observations used for STL and ESD. It must be greater than ``2 * period``.
        max_anomalies (float): Maximum number of anomalies that ESD tests in each window, as a fraction of ``window_size``, greater than 0 and at most 0.49. The test looks for ``max(1, int(max_anomalies * window_size))`` anomalies (Default=0.1).
        alpha (float): ESD significance level, strictly between 0 and 1 (Default=0.05).
        direction (str): Which anomalies the test flags: ``"pos"`` for high values, ``"neg"`` for low values, or ``"both"`` for both, with the critical values of a two-sided test (Default="pos").
        robust (bool): Whether to use robust STL fitting (Default=True).

    Raises:
        ValueError: If ``period`` is less than 2, ``window_size`` is at most ``2 * period``, ``max_anomalies`` is not greater than 0 and at most 0.49, ``alpha`` is not strictly between 0 and 1, or ``direction`` is not ``"pos"``, ``"neg"`` or ``"both"``.
    """

    _use_median = True

    def _center_scale(self, values: np.ndarray) -> tuple[float, float]:
        center = float(np.median(values))
        scale = 1.4826 * float(np.median(np.abs(values - center)))
        return center, scale
