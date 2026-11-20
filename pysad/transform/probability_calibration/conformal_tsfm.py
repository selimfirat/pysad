from __future__ import annotations

import math
from bisect import bisect_left, insort
from typing import cast

import numpy as np

from pysad.core.base_postprocessor import BasePostprocessor
from pysad.utils.window import Window


class ConformalTSFMCalibrator(BasePostprocessor):
    """Online conformal calibrator with adaptive target miscoverage (ACI) :cite:`gibbs2021adaptive`.

    This calibrator implements Adaptive Conformal Inference (ACI) for streaming anomaly scores.
    It maintains a sliding window of recent scores and adapts the target miscoverage level
    ``alpha_t`` based on observed miscoverage events. The calibrator returns the empirical
    p-value of each score (fraction of window scores below it), and exposes the adaptive
    threshold for alerting.

    The ACI update rule is:

    .. math::

        \\alpha_{t+1} = \\alpha_t + \\gamma (\\alpha - \\text{err}_t)

    where ``err_t = 1`` if ``score_t > qhat_t`` else ``0``, and ``qhat_t`` is the
    ``(1 - alpha_t)``-quantile of the recent scores. A momentum variant uses a weighted
    average of past errors:

    .. math::

        \\alpha_{t+1} = \\alpha_t + \\gamma (\\alpha - \\text{weighted_err}_t)

    As for model scores, higher calibrated values mean more anomalous. The calibrated
    score is the fraction of scores in the window that are lower than the input score,
    i.e. one minus its conformal p-value. Alert when the calibrated score exceeds
    ``1 - alpha`` (approximately), or equivalently when the raw score exceeds
    :attr:`threshold_`.

    Args:
        alpha (float): Target miscoverage level in (0, 1). For example, ``alpha=0.1``
            targets 90% coverage. Lower values give wider intervals / higher thresholds.
        gamma (float): Step size for alpha adaptation. Larger values adapt faster to
            distribution shifts but increase volatility.
        window_size (int): Size of the sliding window for score quantiles. Must be at
            least 1 (Default=500).
        method (str): Update method, either ``"simple"`` or ``"momentum"``
            (Default="simple").
        momentum_bw (float): Bandwidth for momentum update, in (0, 1). Closer to 1 gives
            smoother, longer memory (Default=0.95).
        clip_alpha (bool): Whether to clip ``alpha_t`` to ``[0, 1]`` (Default=True).

    Raises:
        ValueError: If ``alpha`` not in (0, 1), ``gamma <= 0``, ``window_size < 1``,
            ``method`` not in {"simple", "momentum"}, or ``momentum_bw`` not in (0, 1).

    Attributes:
        alpha_t (float): Current adaptive miscoverage level ``alpha_t``.
        threshold_ (float): Current adaptive threshold ``qhat_t``, the
            ``(1 - alpha_t)``-quantile of the window.
    """

    __bibtex__ = """
@inproceedings{gibbs2021adaptive,
  title={Adaptive conformal inference under distribution shift},
  author={Gibbs, Isaac and Candès, Emmanuel},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34},
  pages={1660--1672},
  year={2021}
}
"""

    def __init__(
        self,
        alpha: float,
        gamma: float,
        window_size: int = 500,
        method: str = "simple",
        momentum_bw: float = 0.95,
        clip_alpha: bool = True,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")
        if method not in ("simple", "momentum"):
            raise ValueError(f"method must be 'simple' or 'momentum', got '{method}'.")
        if not 0 < momentum_bw < 1:
            raise ValueError("momentum_bw must be in (0, 1).")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.window_size = int(window_size)
        self.method = method
        self.momentum_bw = float(momentum_bw)
        self.clip_alpha = bool(clip_alpha)

        self.window = Window(window_size=self.window_size)
        self._sorted: list[float] = []
        self._alphat = float(alpha)
        self._weighted_err_sum: float = 0.0
        self._weight_sum: float = 0.0
        self._qhat: float = 0.0
        self._has_qhat = False

    @property
    def alpha_t(self) -> float:
        return self._alphat

    @property
    def threshold_(self) -> float:
        return self._qhat

    def _compute_qhat(self) -> float:
        """Compute the (1 - alpha_t)-quantile of the current window."""
        n = len(self.window)
        if n == 0:
            return 0.0
        q_level = float(np.clip(1.0 - self._alphat, 0.0, 1.0))
        # Use the sorted list for efficient quantile computation.
        idx = int(math.ceil(q_level * n)) - 1
        idx = max(0, min(idx, n - 1))
        return float(self._sorted[idx])

    def _update_alpha(self, err_t: float) -> float:
        """Update alpha_t using the ACI update rule."""
        if self.method == "simple":
            self._alphat = self._alphat + self.gamma * (self.alpha - err_t)
        else:
            # Incremental exponential moving average of errors.
            # Weight for new error is 1, previous weights multiplied by momentum_bw.
            self._weighted_err_sum = self.momentum_bw * self._weighted_err_sum + err_t
            self._weight_sum = self.momentum_bw * self._weight_sum + 1.0
            weighted_err = self._weighted_err_sum / self._weight_sum
            self._alphat = self._alphat + self.gamma * (self.alpha - weighted_err)

        if self.clip_alpha:
            self._alphat = float(np.clip(self._alphat, 0.0, 1.0))

        return self._alphat

    def fit_partial(self, score: float) -> ConformalTSFMCalibrator:
        """Fits the next score: computes miscoverage event, updates alpha_t, adds score to window.

        If ``transform_partial`` was called for this score first, it uses the stored ``qhat``.
        Otherwise, it computes ``qhat`` from the window before adding the score.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        s = float(score)
        n = len(self.window)

        if self._has_qhat:
            qhat = self._qhat
            self._has_qhat = False
        else:
            qhat = self._compute_qhat()

        # Miscoverage event: 1 if score exceeds threshold, else 0.
        if self._alphat >= 1.0:
            err_t = 1.0
        elif self._alphat <= 0.0:
            err_t = 0.0
        else:
            err_t = float(s > qhat)

        self._update_alpha(err_t)

        # Add score to window and sorted list.
        if n == self.window_size:
            oldest = float(self.window.window[0])
            if not math.isnan(oldest):
                del self._sorted[bisect_left(self._sorted, oldest)]
        self.window.update(s)
        if not math.isnan(s):
            insort(self._sorted, s)

        return self

    def transform_partial(self, score: float) -> float:
        """Transforms the score using the current window (before adding the score).

        Returns the fraction of scores in the window that are lower than the input score.
        Also computes and stores ``qhat`` for the subsequent ``fit_partial`` call.

        Args:
            score (float): Input score.

        Returns:
            float: Fraction of window scores below the input score, or 0.0 if window is empty.
        """
        n = len(self.window)
        if n == 0 or math.isnan(score):
            self._qhat = 0.0
            self._has_qhat = True
            return 0.0

        # Compute and store qhat for fit_partial.
        self._qhat = self._compute_qhat()
        self._has_qhat = True

        # Return empirical p-value: fraction of window scores < score.
        return bisect_left(self._sorted, float(score)) / n

    def fit_transform_partial(self, score: float) -> float:
        """Transforms the score with the state before the score, then fits the score.

        This matches the standard online workflow: predict/transform using past data,
        then learn from the new observation.

        Args:
            score (float): Input score.

        Returns:
            float: Calibrated score (empirical p-value).
        """
        calibrated = self.transform_partial(score)
        self.fit_partial(score)
        return calibrated

    def fit(self, scores: np.ndarray) -> ConformalTSFMCalibrator:
        """Fits a batch of scores in stream order, keeping the last ``window_size`` scores."""
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
                # Replay alpha_t updates for the tail to match streaming state.
                self._alphat = self.alpha
                self._weighted_err_sum = 0.0
                self._weight_sum = 0.0
                self._has_qhat = False
                for s in tail:
                    qhat = self._compute_qhat()
                    if self._alphat >= 1.0:
                        err_t = 1.0
                    elif self._alphat <= 0.0:
                        err_t = 0.0
                    else:
                        err_t = float(s > qhat)
                    self._update_alpha(err_t)
                return self
            vals = cast(list[float], scores.astype(np.float64, copy=False).tolist())
            for s in vals:
                self.fit_partial(s)
            return self
        super().fit(scores)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Transforms a batch of scores against the frozen calibration window and alpha_t."""
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
                # Transform with current state (before adding s).
                n_curr = len(win_dq)
                if n_curr == 0:
                    out[i] = 0.0
                else:
                    out[i] = _bisect_left(sorted_vals, s) / n_curr

                # Fit: compute qhat from current window, update alpha, add s.
                qhat = self._compute_qhat()
                if self._alphat >= 1.0:
                    err_t = 1.0
                elif self._alphat <= 0.0:
                    err_t = 0.0
                else:
                    err_t = float(s > qhat)
                self._update_alpha(err_t)

                if len(win_dq) == w_size:
                    oldest = float(win_dq[0])
                    if not _isnan(oldest):
                        del sorted_vals[_bisect_left(sorted_vals, oldest)]
                win_dq.append(s)
                _insort(sorted_vals, s)
            return out
        return super().fit_transform(scores)
