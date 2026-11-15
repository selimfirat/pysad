from __future__ import annotations

import math
from collections import deque

import numpy as np

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _int_at_least, check_univariate


class FluxEV(BaseModel):
    """Streaming fluctuation detection with extreme-value thresholds :cite:`li2021fluxev`.

    FluxEV predicts each value with the weighted EWMA in Equation (1).
    It measures the increase in error standard deviation, then subtracts the largest fluctuation from nearby past periods.
    Equations (3)-(7) define these two smoothing steps.
    The score is the resulting nonnegative fluctuation S, before learning the value.
    Higher scores indicate larger unexpected fluctuations.

    The first ``2 * window_size + drift + period * (num_periods - 1)`` values score zero.
    The next ``calibration_size`` fluctuations set the initial 98th percentile threshold.
    Equations (9)-(11) set ``threshold_`` from peak moments and ``risk``.
    After calibration, a score above ``threshold_`` indicates an anomaly.
    Algorithm 3 excludes anomalies from peak updates and from the periodic fluctuation windows.
    It still uses their raw values and prediction errors in later predictions and local smoothing.
    Until calibration ends, ``threshold_`` is infinity.

    Differences from the paper and reference implementation:

    - The paper returns binary labels. This model returns S and exposes the decision threshold as ``threshold_``.
    - Fixed windows and running peak moments replace full histories and the growing peak list. These preserve the equations with bounded memory.
    - The authors' code uses recursive EWMA. This model uses the normalized finite-window weights in Equation (1).
    - The initial percentile uses linear interpolation. The authors' code selects a sorted observation instead.
    - With fewer than two peaks or zero peak variance, the model uses the exponential limit with scale equal to the peak mean.
    - If the peak fraction falls below ``risk``, the decision threshold stays at the initial threshold.
    - With no peaks, the initial threshold remains the decision threshold. The paper does not define these degenerate cases.
    - This model accepts complete, equally spaced values. It does not perform the paper's offline missing-value filling.
    - If every past periodic fluctuation is an anomaly, the periodic baseline is zero.
    - The default period is 24 points. Callers must set it to match their stream.

    The authors publish code at https://github.com/jlidw/FluxEV without a repository license.
    Its SPOT module declares GPLv3. This implementation follows the paper and does not copy that code.
    Memory costs O(window_size + period * num_periods + calibration_size).
    After calibration, each call costs O(window_size + num_periods * (2 * drift + 1)), independent of stream length.

    Args:
        window_size (int): Prediction and local smoothing window s (Default=10).
        period (int): Number of points per period l. It must exceed ``drift`` (Default=24).
        num_periods (int): Periodic window p, including the current period. It must be at least two (Default=7).
        drift (int): Half-width d of each past period's neighborhood (Default=2).
        calibration_size (int): Number k of initial smoothed fluctuations for threshold calibration (Default=1000).
        alpha (float): EWMA weight in (0, 1] (Default=0.2).
        risk (float): Tail probability q in (0, 0.02) (Default=0.001).
    """

    def __init__(
        self,
        window_size: int = 10,
        period: int = 24,
        num_periods: int = 7,
        drift: int = 2,
        calibration_size: int = 1000,
        alpha: float = 0.2,
        risk: float = 0.001,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if num_periods < 2 or not 0 <= drift < period:
            raise ValueError("num_periods must be at least 2 and 0 <= drift < period.")
        if not 0 < alpha <= 1 or not 0 < risk < 0.02:
            raise ValueError("alpha must be in (0, 1] and risk must be in (0, 0.02).")
        self.window_size = window_size
        self.period = period
        self.num_periods = num_periods
        self.drift = drift
        self.calibration_size = _int_at_least(calibration_size, "calibration_size", 1)
        self.alpha = alpha
        self.risk = risk
        weights = (1 - alpha) ** np.arange(window_size - 1, -1, -1, dtype=float)
        self._weights = weights / weights.sum()
        self._values: deque[float] = deque(maxlen=window_size)
        self._errors: deque[float] = deque(maxlen=window_size)
        self._fluctuations = np.full(period * (num_periods - 1) + drift, np.nan)
        # Each row selects one past period and its drift neighborhood.
        self._lags = (
            np.arange(1, num_periods)[:, None] * period + np.arange(-drift, drift + 1)
        ).ravel()
        self._calibration: list[float] = []
        self._count = 0
        self._warmup = 2 * window_size + drift + period * (num_periods - 1)
        self._num_normal = 0
        self._num_peaks = 0
        self._peak_mean = 0.0
        self._peak_m2 = 0.0
        self._initial_threshold = 0.0
        self.threshold_ = math.inf

    def _features(self, x: float) -> tuple[float, float, float]:
        if self._count < self.window_size:
            return 0.0, 0.0, 0.0
        error = x - float(np.dot(self._values, self._weights))
        if self._count < 2 * self.window_size:
            return error, 0.0, 0.0
        errors = np.asarray(self._errors)
        fluctuation = max(float(np.append(errors, error).std() - errors.std()), 0.0)
        if self._count < self._warmup:
            return error, fluctuation, 0.0
        slots = (self._count - self._lags) % self._fluctuations.size
        baseline = float(np.fmax.reduce(self._fluctuations[slots], initial=0.0))
        return error, fluctuation, max(fluctuation - baseline, 0.0)

    def _add_peak(self, peak: float) -> None:
        self._num_peaks += 1
        delta = peak - self._peak_mean
        self._peak_mean += delta / self._num_peaks
        self._peak_m2 += delta * (peak - self._peak_mean)

    def _update_threshold(self) -> None:
        self.threshold_ = self._initial_threshold
        if not self._num_peaks:
            return
        mean = self._peak_mean
        variance = self._peak_m2 / (self._num_peaks - 1) if self._num_peaks > 1 else 0.0
        shape, scale = 0.0, mean
        if variance > 0:
            ratio = mean * mean / variance
            shape = (1 - ratio) / 2
            scale = mean * (1 + ratio) / 2
        log_ratio = math.log(self._num_normal * self.risk / self._num_peaks)
        if log_ratio >= 0:
            return
        if abs(shape) < 1e-8:
            excess = -scale * log_ratio
        else:
            excess = scale * math.expm1(-shape * log_ratio) / shape
        self.threshold_ += max(excess, 0.0)

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> FluxEV:
        """Updates the windows and threshold with one value.

        Args:
            X (np.ndarray): One value with shape (1,).
            y (int or None): Ignored label (Default=None).

        Returns:
            FluxEV: The model itself.
        """
        x = check_univariate(X, "FluxEV")
        error, fluctuation, score = self._features(x)
        alarm = score > self.threshold_
        if self._count >= self._warmup:
            if self._num_normal == 0:
                self._calibration.append(score)
                if len(self._calibration) == self.calibration_size:
                    self._initial_threshold = float(np.quantile(self._calibration, 0.98))
                    self._num_normal = self.calibration_size
                    for value in self._calibration:
                        if value > self._initial_threshold:
                            self._add_peak(value - self._initial_threshold)
                    self._calibration.clear()
                    self._update_threshold()
            elif not alarm:
                self._num_normal += 1
                if score > self._initial_threshold:
                    self._add_peak(score - self._initial_threshold)
                    self._update_threshold()
        self._values.append(x)
        if self._count >= self.window_size:
            self._errors.append(error)
        if self._count >= 2 * self.window_size:
            self._fluctuations[self._count % self._fluctuations.size] = (
                math.nan if alarm else fluctuation
            )
        self._count += 1
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Returns the smoothed fluctuation without changing the model.

        Args:
            X (np.ndarray): One value with shape (1,).

        Returns:
            float: The nonnegative fluctuation S, or zero during the smoothing warmup.
        """
        return self._features(check_univariate(X, "FluxEV"))[2]
