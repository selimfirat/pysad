from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t

from pysad.core.base_model import BaseModel
from pysad.statistics import OnlinePeriodicityMeter
from pysad.transform.preprocessing import ModifiedSTLResidualTransformer
from pysad.utils.validation import check_univariate
from pysad.utils.window import RingBuffer, _median_and_mad_sorted


@lru_cache(maxsize=1024)
def _critical_value(alpha: float, num_tails: float, n: int, i: int) -> float:
    # The critical value of the i-th ESD test on n values. It depends only on these numbers, so it
    # is computed once instead of once per scored instance.
    p = 1.0 - alpha / (num_tails * (n - i + 1))
    t_value = t.ppf(p, n - i - 1)
    return float(((n - i) * t_value) / np.sqrt((n - i - 1 + t_value**2) * (n - i + 1)))


def _median_mad_sorted(sub: NDArray[np.float64]) -> tuple[float, float]:
    """Returns ``(median, 1.4826 * MAD)`` of sorted 1-D slice ``sub`` in O(log m) time."""
    center, mad = _median_and_mad_sorted(sub)
    return center, 1.4826 * mad


class SeasonalESD(BaseModel):
    """Flags seasonal anomalies in a univariate stream with the Seasonal ESD (S-ESD) test over a sliding window :cite:`hochenbaum2017automatic`.

    This is the S-ESD method of the paper (Eq. 5). It computes the modified STL residual ``X - seasonal - median(X)`` over a fixed-size window, with the periodic STL of R (see :class:`pysad.transform.preprocessing.ModifiedSTLResidualTransformer`), then applies generalized ESD with the mean and the standard deviation to the residuals. The reference implementation of the paper, Twitter's AnomalyDetection R package, has no S-ESD: it always uses the median and the MAD, as :class:`SeasonalHybridESD` does. This model follows the R package in everything else. The test stops when the scale of the remaining values is at most ``window_size`` times the float64 machine epsilon times the largest absolute value in the window, a bound on the rounding error of the residuals, where R stops only when the MAD is exactly 0, so that the floating-point noise of a series without anomalies raises no alerts. The paper runs the test on a full batch. This model runs it on the most recent ``window_size`` values for each instance.

    The score of an instance is the ESD test statistic of the newest value if the test flags it as an anomaly, else 0.0. The score is 0.0 until the window holds ``window_size`` values.

    When ``auto_period=True``, the model estimates the period online using :class:`pysad.statistics.OnlinePeriodicityMeter` and updates the seasonal decomposition when the estimated period changes. The period meter uses a sliding window of ``period_window_size`` observations (defaulting to ``window_size``). The initial ``period`` is used until the meter produces its first valid estimate.

    Args:
        period (int | None): Number of observations in one seasonal period. It must be at least 2.
            Optional when ``auto_period=True``; required otherwise.
        window_size (int): Number of recent observations used for STL and ESD.
            When ``auto_period=False``, it must be greater than ``2 * period``.
            When ``auto_period=True``, it must be greater than ``2 * min_period``.
        max_anomalies (float): Maximum number of anomalies that ESD tests in each window,
            as a fraction of ``window_size``, greater than 0 and at most 0.49.
            The test looks for ``max(1, int(max_anomalies * window_size))`` anomalies (Default=0.1).
        alpha (float): ESD significance level, strictly between 0 and 1 (Default=0.05).
        direction (str): Which anomalies the test flags: ``"pos"`` for high values,
            ``"neg"`` for low values, or ``"both"`` for both, with the critical values of a
            two-sided test (Default="pos").
        robust (bool): Whether to use robust STL fitting (Default=True).
        auto_period (bool): Whether to estimate the period online with
            :class:`pysad.statistics.OnlinePeriodicityMeter` (Default=False).
        period_window_size (int | None): Window size for the period meter.
            Defaults to ``window_size``. Must be at least 4.
        min_period (int): Minimum period to accept from the meter (Default=2).
        max_period (int | None): Maximum period to accept from the meter.
            Defaults to ``period_window_size // 2`` (the Nyquist limit).

    Raises:
        ValueError: If validation fails for the provided parameters.
    """

    _use_median = False

    @staticmethod
    def _resolve_period_params(
        period: int | None,
        window_size: int,
        auto_period: bool,
        period_window_size: int | None,
        min_period: int,
        max_period: int | None,
    ) -> tuple[int, int, int]:
        if min_period < 2:
            raise ValueError("min_period must be at least 2.")
        if period_window_size is not None and period_window_size < 4:
            raise ValueError("period_window_size must be at least 4.")

        resolved_pws = period_window_size if period_window_size is not None else window_size
        resolved_max = max_period if max_period is not None else resolved_pws // 2
        if auto_period:
            resolved_period = min_period if period is None else period
            if window_size <= 2 * min_period:
                raise ValueError(
                    "window_size must be greater than 2 * min_period when auto_period=True."
                )
            if resolved_max < min_period:
                raise ValueError("max_period must be at least min_period.")
            return resolved_period, resolved_pws, resolved_max

        if period is None:
            raise ValueError("period is required when auto_period=False.")
        if period < 2:
            raise ValueError("period must be at least 2.")
        if window_size <= 2 * period:
            raise ValueError("window_size must be greater than 2 * period.")
        return period, resolved_pws, resolved_max

    def __init__(
        self,
        period: int | None = None,
        window_size: int = 100,
        max_anomalies: float = 0.1,
        alpha: float = 0.05,
        direction: str = "pos",
        robust: bool = True,
        auto_period: bool = False,
        period_window_size: int | None = None,
        min_period: int = 2,
        max_period: int | None = None,
    ) -> None:
        if not 0 < max_anomalies <= 0.49:
            raise ValueError("max_anomalies must be greater than 0 and at most 0.49.")
        if not np.isfinite(alpha) or not (0 < alpha < 1):
            raise ValueError("alpha must be finite and between 0 and 1.")
        if direction not in ("pos", "neg", "both"):
            raise ValueError("direction must be 'pos', 'neg' or 'both'.")

        resolved_period, resolved_pws, resolved_max = self._resolve_period_params(
            period, window_size, auto_period, period_window_size, min_period, max_period
        )

        self.auto_period = auto_period
        self.period_window_size = resolved_pws
        self.min_period = min_period
        self.max_period = resolved_max
        self.period = resolved_period
        self.window_size = window_size
        self.max_anomalies = max_anomalies
        self.alpha = alpha
        self.direction = direction
        self._window = RingBuffer(window_size)
        self.residual_transformer = ModifiedSTLResidualTransformer(
            period=resolved_period, window_size=window_size, robust=robust
        )
        self._period_meter: OnlinePeriodicityMeter | None = (
            OnlinePeriodicityMeter(window_size=resolved_pws) if auto_period else None
        )

    def _as_value(self, X: NDArray[np.float64]) -> float:
        return check_univariate(X, "SeasonalESD")

    def _center_scale(self, values: NDArray[np.float64]) -> tuple[float, float]:
        return float(values.mean()), float(values.std(ddof=1))

    def _critical_value(self, n: int, i: int) -> float:
        return _critical_value(self.alpha, 2.0 if self.direction == "both" else 1.0, n, i)

    @staticmethod
    def _build_shifted_prefix_sums(
        s: NDArray[np.float64], mid: int
    ) -> tuple[
        float,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        n = s.shape[0]
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
        return ref, left_s1, right_s1, left_s2, right_s2

    @staticmethod
    def _select_extreme(
        direction: str,
        s: NDArray[np.float64],
        orig_idx: NDArray[np.intp],
        lo: int,
        hi: int,
        center: float,
        scale: float,
    ) -> tuple[bool, float]:
        if direction == "pos":
            return False, (float(s[hi]) - center) / scale
        if direction == "neg":
            return True, (center - float(s[lo])) / scale
        stat_lo = abs(float(s[lo]) - center) / scale
        stat_hi = abs(float(s[hi]) - center) / scale
        if stat_lo != stat_hi:
            return (True, stat_lo) if stat_lo > stat_hi else (False, stat_hi)
        min_lo = int(orig_idx[lo])
        if s[hi - 1] == s[hi]:
            p_tie = lo + int(np.searchsorted(s[lo : hi + 1], s[hi], side="left"))
            min_hi = int(orig_idx[p_tie])
        else:
            min_hi = int(orig_idx[hi])
        return min_lo < min_hi, stat_lo

    @staticmethod
    def _peel_extreme(
        orig_idx: NDArray[np.intp],
        s: NDArray[np.float64],
        lo: int,
        hi: int,
        pick_low: bool,
    ) -> tuple[int, int, int]:
        if pick_low:
            return int(orig_idx[lo]), lo + 1, hi
        if hi > lo and s[hi - 1] == s[hi]:
            p = lo + int(np.searchsorted(s[lo : hi + 1], s[hi], side="left"))
            peeled_idx = int(orig_idx[p])
            orig_idx[p:hi] = orig_idx[p + 1 : hi + 1]
            return peeled_idx, lo, hi - 1
        return int(orig_idx[hi]), lo, hi - 1

    def _window_center_scale(
        self,
        s: NDArray[np.float64],
        lo: int,
        hi: int,
        mid: int,
        ref: float,
        left_s1: NDArray[np.float64],
        right_s1: NDArray[np.float64],
        left_s2: NDArray[np.float64],
        right_s2: NDArray[np.float64],
    ) -> tuple[float, float]:
        if self._use_median:
            return _median_mad_sorted(s[lo : hi + 1])
        m = hi - lo + 1
        if lo <= mid <= hi + 1:
            sum1 = float(left_s1[lo] + right_s1[hi + 1 - mid])
            sum2 = float(left_s2[lo] + right_s2[hi + 1 - mid])
            mean_shifted = sum1 / m
            center = ref + mean_shifted
            var = (sum2 - sum1 * mean_shifted) / (m - 1)
            scale = float(np.sqrt(var)) if var > 0.0 else 0.0
            return center, scale
        return self._center_scale(s[lo : hi + 1])

    def _esd_target(self, values: NDArray[np.float64], min_scale: float, target_idx: int) -> float:
        arr = np.asarray(values, dtype=np.float64)
        n = arr.shape[0]
        max_k = max(1, int(self.max_anomalies * n))
        orig_idx = np.argsort(arr, kind="stable")
        s = arr[orig_idx]

        mid = n >> 1
        if self._use_median:
            ref = 0.0
            left_s1 = right_s1 = left_s2 = right_s2 = arr
        else:
            ref, left_s1, right_s1, left_s2, right_s2 = self._build_shifted_prefix_sums(s, mid)

        lo = 0
        hi = n - 1
        direction = self.direction
        target_stat: float | None = None

        for i in range(1, max_k + 1):
            if hi - lo + 1 <= 1:
                break
            center, scale = self._window_center_scale(
                s, lo, hi, mid, ref, left_s1, right_s1, left_s2, right_s2
            )
            if scale <= min_scale:
                break

            pick_low, statistic = self._select_extreme(
                direction, s, orig_idx, lo, hi, center, scale
            )
            peeled_idx, lo, hi = self._peel_extreme(orig_idx, s, lo, hi, pick_low)

            critical_value = self._critical_value(n, i)
            if peeled_idx == target_idx:
                if statistic > critical_value:
                    return float(statistic)
                target_stat = float(statistic)
            elif target_stat is not None and statistic > critical_value:
                return target_stat

        return 0.0

    def fit_partial(self, X: NDArray[np.float64], y: int | None = None) -> SeasonalESD:
        """Adds the next instance to the model window.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.
            y (int or None): Ignored. Present for API consistency (Default=None).

        Returns:
            object: self.
        """
        value = self._as_value(X)
        self._window.append(value)

        if self.auto_period and self._period_meter is not None:
            self._period_meter.update(value)
            estimated_period = self._period_meter.get()
            if estimated_period > 0:
                new_period = int(round(estimated_period))
                if self.min_period <= new_period <= self.max_period and new_period != self.period:
                    if self.window_size > 2 * new_period:
                        self.period = new_period
                        self.residual_transformer = ModifiedSTLResidualTransformer(
                            period=new_period,
                            window_size=self.window_size,
                            robust=self.residual_transformer.robust,
                        )

        return self

    def _score_window(self, values: NDArray[np.float64]) -> float:
        residuals = self.residual_transformer.transform_window(values)
        # The rounding error of the residuals grows with the window size and the largest value.
        return self._esd_target(
            residuals,
            min_scale=float(
                self.window_size * np.finfo(np.float64).eps * float(np.abs(values).max())
            ),
            target_idx=values.shape[0] - 1,
        )

    def score_partial(self, X: NDArray[np.float64]) -> float:
        """Scores the next instance in the window that ends with it, without adding it to the window.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.

        Returns:
            float: The ESD test statistic of ``X`` if the test flags it as an anomaly, else 0.0.
            Higher scores represent more anomalous instances.
        """
        value = self._as_value(X)
        if len(self._window) + 1 < self.window_size:
            return 0.0

        # The window without the oldest item (which would be evicted when the new item arrives).
        window = self._window.window_view()
        if len(self._window) == self.window_size:
            window = window[:-1]
        candidate = np.append(window, value)
        return self._score_window(candidate)

    def fit_score_partial(self, X: NDArray[np.float64], y: int | None = None) -> float:
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
        return self._score_window(self._window.window_view())


class SeasonalHybridESD(SeasonalESD):
    """Flags seasonal anomalies in a univariate stream with the Seasonal Hybrid ESD (S-H-ESD) test over a sliding window :cite:`hochenbaum2017automatic`.

    This is the S-H-ESD method of the paper. It uses the same periodic modified STL residual as :class:`SeasonalESD`, then replaces the mean and the standard deviation of ESD with the median and the MAD-based scale (1.4826 times the MAD) in the test statistic. For each window, the test flags the anomalies that ``AnomalyDetectionVec(window, max_anoms = max_anomalies, direction = direction, alpha = alpha, period = period)`` of Twitter's AnomalyDetection R package, the reference implementation of the paper, flags. The exception is an even ``window_size``, for which the robust STL sometimes differs from R, as :class:`pysad.transform.preprocessing.ModifiedSTLResidualTransformer` documents. Use an odd ``window_size`` to match R. The test stops when the scale of the remaining values is at most ``window_size`` times the float64 machine epsilon times the largest absolute value in the window, a bound on the rounding error of the residuals, where R stops only when the MAD is exactly 0, so that the floating-point noise of a series without anomalies raises no alerts. The paper runs the test on a full batch. This model runs it on the most recent ``window_size`` values for each instance.

    The score of an instance is the test statistic of the newest value if the test flags it as an anomaly, else 0.0. The score is 0.0 until the window holds ``window_size`` values.

    When ``auto_period=True``, the model estimates the period online using :class:`pysad.statistics.OnlinePeriodicityMeter` and updates the seasonal decomposition when the estimated period changes. The period meter uses a sliding window of ``period_window_size`` observations (defaulting to ``window_size``). The initial ``period`` is used until the meter produces its first valid estimate.

    Args:
        period (int | None): Number of observations in one seasonal period. It must be at least 2.
            Optional when ``auto_period=True``; required otherwise.
        window_size (int): Number of recent observations used for STL and ESD.
            When ``auto_period=False``, it must be greater than ``2 * period``.
            When ``auto_period=True``, it must be greater than ``2 * min_period``.
        max_anomalies (float): Maximum number of anomalies that ESD tests in each window,
            as a fraction of ``window_size``, greater than 0 and at most 0.49.
            The test looks for ``max(1, int(max_anomalies * window_size))`` anomalies (Default=0.1).
        alpha (float): ESD significance level, strictly between 0 and 1 (Default=0.05).
        direction (str): Which anomalies the test flags: ``"pos"`` for high values,
            ``"neg"`` for low values, or ``"both"`` for both, with the critical values of a
            two-sided test (Default="pos").
        robust (bool): Whether to use robust STL fitting (Default=True).
        auto_period (bool): Whether to estimate the period online with
            :class:`pysad.statistics.OnlinePeriodicityMeter` (Default=False).
        period_window_size (int | None): Window size for the period meter.
            Defaults to ``window_size``. Must be at least 4.
        min_period (int): Minimum period to accept from the meter (Default=2).
        max_period (int | None): Maximum period to accept from the meter.
            Defaults to ``period_window_size // 2`` (the Nyquist limit).

    Raises:
        ValueError: If validation fails for the provided parameters.
    """

    _use_median = True

    def _center_scale(self, values: NDArray[np.float64]) -> tuple[float, float]:
        center = float(np.median(values))
        scale = 1.4826 * float(np.median(np.abs(values - center)))
        return center, scale
