from __future__ import annotations

import math
from bisect import bisect_left, insort
from collections import OrderedDict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pysad.core.base_postprocessor import BasePostprocessor
from pysad.utils.window import Window


class GroupConditionalConformalCalibrator(BasePostprocessor):
    """Regime-stratified conformal calibrator with conditional coverage per regime :cite:`gupta2020distributionfree,barber2021predictive`.

    The calibrator keeps one sliding calibration window per regime (group). A score is transformed
    against its own regime's window: it returns the fraction of that window's scores below the
    score, i.e. one minus the score's conformal p-value within that regime. Stratifying by regime
    gives conditional coverage per regime instead of only marginal coverage over the whole stream,
    as in distribution-free binary classification with prediction sets and conditional coverage.

    With ``regime=None`` (the default) all scores share one window, so the calibrator behaves
    exactly like :class:`~pysad.transform.probability_calibration.ConformalProbabilityCalibrator`.
    Pass a regime, for example an operating mode or a drift segment, to calibrate each regime
    separately.

    As for model scores, higher values mean more anomalous. Alert when the calibrated score
    exceeds ``1 - alpha`` for a target miscoverage ``alpha``.

    Each regime window holds at most ``window_size`` scores. The number of tracked regimes is
    bounded by ``max_regimes``: when a new regime would exceed it, the least recently used regime
    is evicted, so memory stays bounded no matter how many regimes the stream produces.

    The paper derives distribution-free calibration from score level sets. This calibrator
    approximates that with per-regime empirical CDFs over sliding windows. Exact conditional
    coverage requires exchangeability within a regime; the streaming setting approximates it. When
    no regime is given, the result matches the marginal (single-window) conformal calibrator.

    Args:
        window_size (int): Scores kept per regime. Must be at least 1 (Default=500).
        max_regimes (int): Maximum number of regimes tracked at once. Must be at least 1
            (Default=64).

    Raises:
        ValueError: If ``window_size < 1`` or ``max_regimes < 1``.

    Attributes:
        window_size (int): Scores kept per regime.
        max_regimes (int): Maximum number of regimes tracked at once.
    """

    __bibtex__ = """
@article{gupta2020distributionfree,
  title={Distribution-free binary classification: prediction sets, confidence intervals and calibration},
  author={Gupta, Chirag and Podkopaev, Aleksandr and Ramdas, Aaditya},
  journal={arXiv preprint arXiv:2006.10564},
  year={2020},
  doi={10.48550/arXiv.2006.10564}
}

@article{barber2021predictive,
  title={Predictive inference with the jackknife+},
  author={Barber, Rina Foygel and Cand{\\`e}s, Emmanuel J. and Ramdas, Aaditya and Tibshirani, Ryan J.},
  journal={The Annals of Statistics},
  volume={49},
  number={2},
  pages={486--507},
  year={2021},
  doi={10.1214/20-AOS1965}
}
"""

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

    def fit_partial(self, score: float, regime: Any = None) -> GroupConditionalConformalCalibrator:
        """Fits the next score into its regime's window without changing the calibrated output.

        The score is added to the regime's window after it is transformed, matching the online
        workflow where a score is evaluated against past data before the model learns from it.
        When the regime window is full, its oldest score is evicted.

        Args:
            score (float): Input score.
            regime: Regime or group the score belongs to. ``None`` uses the shared default window.

        Returns:
            object: self.
        """
        s = float(score)
        window, sorted_vals = self._get_or_create(regime)

        if len(window) == self.window_size:
            oldest = float(window.window[0])
            if not math.isnan(oldest):
                del sorted_vals[bisect_left(sorted_vals, oldest)]
        window.update(s)
        if not math.isnan(s):
            insort(sorted_vals, s)

        return self

    def fit_transform_partial(self, score: float, regime: Any = None) -> float:
        """Updates the calibrator with the next score and transforms the score.

        The score is transformed against its regime's window before it is added, matching the
        online workflow where a score is evaluated against past data before the model learns from
        it.

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
        s = float(score)
        if math.isnan(s):
            return 0.0
        entry = self._regimes.get(regime)
        if entry is None:
            return 0.0
        window, sorted_vals = entry
        n = len(window)
        if n == 0 or not sorted_vals:
            return 0.0
        return bisect_left(sorted_vals, s) / n

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
            win_dq = window.window
            w_size = self.window_size
            _bisect_left = bisect_left
            _insort = insort
            _isnan = math.isnan
            for i, s in enumerate(scores.tolist()):
                if len(win_dq) == w_size:
                    oldest = float(win_dq[0])
                    if not _isnan(oldest):
                        del sorted_vals[_bisect_left(sorted_vals, oldest)]
                win_dq.append(s)
                _insort(sorted_vals, s)
                out[i] = _bisect_left(sorted_vals, s) / len(win_dq)
            return out
        return super().fit_transform(scores)
