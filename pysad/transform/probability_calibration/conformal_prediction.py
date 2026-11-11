from __future__ import annotations

import math
from bisect import bisect_left, insort
from typing import cast

import numpy as np

from pysad.core.base_postprocessor import BasePostprocessor
from pysad.utils.window import Window


class ConformalProbabilityCalibrator(BasePostprocessor):
    """Converts each score into the fraction of scores in the window that are lower than it, by conformal prediction :cite:`laxhammar2013online,ishimtsev2017conformal`.

    The calibrator works on the scores that the model gave when each instance arrived. This follows the sliding calibration queue of Ishimtsev et al. (2017) (LDCD), where the window holds the scores that the model gave when each instance arrived, with the test score included. When ``window_size`` is the queue length plus one, the calibrator returns one minus that conformal p-value. In contrast, Conformal Anomaly Detection (CAD) :cite:`laxhammar2013online` recomputes the nonconformity scores of all previous instances against the current training set for each new instance.

    This calibrator transforms a score into the fraction of scores in the window that are lower than it, i.e. one minus its conformal p-value. As for model scores, higher values mean more anomalous, e.g. alert when the calibrated score is above 0.95, which is a conformal p-value below 0.05.

    The target score is assumed to be fitted before it is transformed (e.g. via ``fit_transform_partial``), so it is counted in the window. For a score that is not in the window, the exact conformal p-value would be (count + 1) / (n + 1), where count is the number of scores in the window that are greater than or equal to it.

    Args:
        window_size (int): The number of scores kept in the window. Must be at least 1 (Default=300).

    Raises:
        ValueError: If window_size is less than 1.
    """

    def __init__(self, window_size: int = 300) -> None:
        self.window_size = window_size
        self.window = Window(window_size=self.window_size)
        self._sorted: list[float] = []

    def fit_partial(self, score: float) -> ConformalProbabilityCalibrator:
        """Fits particular (next) timestep's score to train the postprocessor.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        if len(self.window) == self.window_size:
            oldest = float(self.window.window[0])
            if not math.isnan(oldest):
                del self._sorted[bisect_left(self._sorted, oldest)]
        self.window.update(score)
        s = float(score)
        if not math.isnan(s):
            insort(self._sorted, s)

        return self

    def transform_partial(self, score: float) -> float:
        """Transforms given score.

        Args:
            score (float): Input score.

        Returns:
            float: One minus the conformal p-value of the score, or 0.0 before the first fit.
        """
        n = len(self.window)
        if n == 0 or math.isnan(score):
            return 0.0
        return bisect_left(self._sorted, float(score)) / n

    def fit(self, scores: np.ndarray) -> ConformalProbabilityCalibrator:
        """Fits all scores in stream order, keeping the last ``window_size`` scores."""
        if isinstance(scores, np.ndarray) and scores.ndim == 1:
            if len(scores) == 0:
                return self
            if len(scores) >= self.window_size:
                tail = cast(
                    list[float], scores[-self.window_size :].astype(np.float64, copy=False).tolist()
                )
                self.window.window.clear()
                self.window.window.extend(tail)
                self._sorted = sorted(x for x in tail if not math.isnan(x))
                return self
            vals = cast(list[float], scores.astype(np.float64, copy=False).tolist())
            for s in vals:
                self.fit_partial(s)
            return self
        super().fit(scores)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Transforms a batch of scores against the frozen calibration window."""
        arr = np.asarray(scores, dtype=np.float64)
        if arr.ndim != 1:
            return super().transform(scores)
        n = len(self.window)
        if len(arr) == 0 or n == 0 or not self._sorted:
            return np.zeros(len(arr), dtype=np.float64)
        sorted_arr = np.asarray(self._sorted, dtype=np.float64)
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            safe = np.where(nan_mask, 0.0, arr)
            out = np.searchsorted(sorted_arr, safe, side="left") / n
            out[nan_mask] = 0.0
            return out
        return np.searchsorted(sorted_arr, arr, side="left") / n

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        """Fits and calibrates a 1-D score batch in ``O(N log W)`` time."""
        if isinstance(scores, np.ndarray) and scores.ndim == 1 and not np.isnan(scores).any():
            m = len(scores)
            out = np.empty(m, dtype=np.float64)
            if m == 0:
                return out
            win_dq = self.window.window
            sorted_vals = self._sorted
            w_size = self.window_size
            _bisect_left = bisect_left
            _insort = insort
            _isnan = math.isnan
            vals = cast(list[float], scores.astype(np.float64, copy=False).tolist())
            for i, s in enumerate(vals):
                if len(win_dq) == w_size:
                    oldest = float(win_dq[0])
                    if not _isnan(oldest):
                        del sorted_vals[_bisect_left(sorted_vals, oldest)]
                win_dq.append(s)
                _insort(sorted_vals, s)
                out[i] = _bisect_left(sorted_vals, s) / len(win_dq)
            return out
        return super().fit_transform(scores)
