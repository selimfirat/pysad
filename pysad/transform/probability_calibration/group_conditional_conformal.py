from __future__ import annotations

import math
from bisect import bisect_left, insort
from collections import OrderedDict
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from pysad.core.base_postprocessor import BasePostprocessor
from pysad.utils.window import Window


class GroupConditionalConformalCalibrator(BasePostprocessor):
    """Regime-stratified conformal calibrator with one bounded window per regime :cite:`laxhammar2013online,ishimtsev2017conformal`.

    The calibrator keeps one sliding calibration window per regime (group). A score is transformed
    against its own regime's window: it returns the fraction of that window's scores below the
    score, i.e. one minus the score's conformal p-value within that regime. This rank transform
    follows the online conformal anomaly detection of Laxhammar and Falkman (2014) and the sliding calibration
    queue of Ishimtsev et al. (2017).

    Stratifying by regime gives conditional coverage per regime instead of only marginal coverage
    over the whole stream. With ``regime=None`` (the default) all scores share one window, so the
    calibrator behaves exactly like :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator`.
    Pass a regime, for example an operating mode or a drift segment, to calibrate each regime
    separately.

    As for model scores, higher values mean more anomalous. Alert when the calibrated score
    exceeds ``1 - alpha`` for a target miscoverage ``alpha``.

    Each regime window holds at most ``window_size`` scores. The number of tracked regimes is
    bounded by ``max_regimes``: when a new regime would exceed it, the least recently used regime
    is evicted, so memory stays bounded no matter how many regimes the stream produces.

    No cited paper describes this full regime-stratified sliding-window conformal rank transform.
    The cited papers describe per-instance conformal rank transforms without regime stratification.
    The sliding windows, LRU eviction, and regime cap are PySAD additions. Conditional coverage is
    exact only under exchangeability within a regime; the streaming setting approximates it.

    Args:
        window_size (int): Scores kept per regime. Must be at least 1 (Default=500).
        max_regimes (int): Maximum number of regimes tracked at once. Must be at least 1
            (Default=64).

    Raises:
        ValueError: If ``window_size < 1`` or ``max_regimes < 1``.
        TypeError: If ``score`` is not convertible to float.

    Attributes:
        window_size (int): Scores kept per regime.
        max_regimes (int): Maximum number of regimes tracked at once.
    """

    _output_kind: ClassVar[Literal["score", "probability", "decision"]] = "probability"

    def __init__(self, window_size: int = 500, max_regimes: int = 64) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if max_regimes < 1:
            raise ValueError("max_regimes must be at least 1.")

        self.window_size = int(window_size)
        self.max_regimes = int(max_regimes)

        # Regime -> (Window, sorted scores). OrderedDict gives least-recently-used eviction order.
        self._regimes: OrderedDict[Any, tuple[Window, list[float]]] = OrderedDict()

    def _get_or_create(self, regime: Any) -> tuple[Window, list[float]]:
        """Returns the regime's window and sorted scores, creating it when it is missing."""
        entry = self._regimes.get(regime)
        if entry is None:
            if len(self._regimes) >= self.max_regimes:
                self._regimes.popitem(last=False)
            entry = (Window(window_size=self.window_size), [])
            self._regimes[regime] = entry
        self._regimes.move_to_end(regime)
        return entry

    def _validate_score(self, score: float) -> float:
        """Convert score to float, raising TypeError if not possible."""
        try:
            return float(score)
        except (TypeError, ValueError):
            raise TypeError(
                f"score must be convertible to float, got {type(score).__name__}: {score!r}"
            ) from None

    def fit_partial(self, score: float, regime: Any = None) -> GroupConditionalConformalCalibrator:
        """Adds the score to its regime's window without transforming it.

        fit_partial does not transform the score; it only adds it to the regime's window. Adding
        the score changes future calibrated output, because later scores are compared against it.
        When the regime window is full, its oldest score is evicted.

        Args:
            score (float): Input score.
            regime: Regime or group the score belongs to. ``None`` uses the shared default window.

        Returns:
            object: self.
        """
        score = self._validate_score(score)
        window, sorted_vals = self._get_or_create(regime)

        if len(window) == self.window_size:
            oldest = window._buffer.oldest()
            if oldest is not None and not math.isnan(oldest):
                del sorted_vals[bisect_left(sorted_vals, oldest)]
        window.update(score)
        if not math.isnan(score):
            insort(sorted_vals, score)

        return self

    def fit_transform_partial(self, score: float, regime: Any = None) -> float:
        """Adds the score to its regime's window, then transforms it.

        The score is added to the window before it is transformed, so it is counted in its own
        fraction. The score is transformed against its regime's window after it is added.

        Args:
            score (float): Input score.
            regime: Regime or group the score belongs to. ``None`` uses the shared default window.

        Returns:
            float: Calibrated score.
        """
        self.fit_partial(score, regime=regime)
        return self.transform_partial(score, regime=regime)

    def transform_partial(self, score: float, regime: Any = None) -> float:
        """Transforms the score against its regime's window without changing the state.

        Returns the fraction of the regime's window scores below the score, or ``0.0`` when the
        regime has no window yet, is empty, or the score is missing.

        Args:
            score (float): Input score.
            regime: Regime or group the score belongs to. ``None`` uses the shared default window.

        Returns:
            float: Fraction of the regime's window scores below the score, or ``0.0`` if the regime
                has no scores yet or the score is missing.
        """
        s = self._validate_score(score)
        if math.isnan(s):
            return 0.0
        entry = self._regimes.get(regime)
        if entry is None:
            return 0.0
        window, sorted_vals = entry
        n = len(window)
        if n == 0 or not sorted_vals:
            return 0.0
        return bisect_left(sorted_vals, float(s)) / n

    def transform(self, scores: NDArray[Any]) -> NDArray[Any]:
        """Transforms a batch of scores against the frozen default-regime window.

        Batch calibration only supports the default ``regime=None``. Use ``transform_partial`` with
        a regime for per-regime calibration.

        Args:
            scores (np.float64 array of shape (num_instances,)): Input scores.

        Returns:
            np.float64 array of shape (num_instances,): Calibrated scores.
        """
        arr = np.asarray(scores, dtype=np.float64)
        if arr.ndim != 1:
            return super().transform(scores)
        n_out = len(arr)
        if n_out == 0:
            return np.empty(0, dtype=np.float64)
        entry = self._regimes.get(None)
        if entry is None:
            return np.zeros(n_out, dtype=np.float64)
        window, sorted_vals = entry
        n = len(window)
        if n == 0 or not sorted_vals:
            return np.zeros(n_out, dtype=np.float64)
        sorted_arr = np.asarray(sorted_vals, dtype=np.float64)
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            safe = np.where(nan_mask, 0.0, arr)
            out = np.searchsorted(sorted_arr, safe, side="left") / n
            out[nan_mask] = 0.0
            return out
        return np.searchsorted(sorted_arr, arr, side="left") / n

    def fit_transform(self, scores: NDArray[Any]) -> NDArray[Any]:
        """Fits and calibrates a 1-D score batch in ``O(N log W)`` time for the default regime.

        Batch calibration only supports the default ``regime=None``. Missing values fall back to
        the per-score implementation.

        Args:
            scores (np.float64 array of shape (num_instances,)): Input scores.

        Returns:
            np.float64 array of shape (num_instances,): Calibrated scores.
        """
        if isinstance(scores, np.ndarray) and scores.ndim == 1 and not np.isnan(scores).any():
            m = len(scores)
            out = np.empty(m, dtype=np.float64)
            if m == 0:
                return out
            window, sorted_vals = self._get_or_create(None)
            buf = window._buffer
            w_size = self.window_size
            _bisect_left = bisect_left
            _insort = insort
            _isnan = math.isnan
            for i, s in enumerate(scores.tolist()):
                if len(buf) == w_size:
                    oldest = buf.oldest()
                    if oldest is not None and not _isnan(oldest):
                        del sorted_vals[_bisect_left(sorted_vals, oldest)]
                buf.append(s)
                _insort(sorted_vals, s)
                out[i] = _bisect_left(sorted_vals, s) / len(buf)
            return out
        return super().fit_transform(scores)
