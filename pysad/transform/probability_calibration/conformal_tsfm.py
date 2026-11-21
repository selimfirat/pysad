from __future__ import annotations

import math
from bisect import bisect_left, insort
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from pysad.core.base_postprocessor import BasePostprocessor
from pysad.utils.window import Window


class ConformalTSFMCalibrator(BasePostprocessor):
    """Online conformal calibrator with adaptive target miscoverage (ACI) :cite:`gibbs2021adaptive`.

    This calibrator implements Adaptive Conformal Inference (ACI) for streaming anomaly scores.
    It keeps a sliding window of recent scores and adapts the target miscoverage level
    ``alpha_t`` based on the miscoverage events it observes. The calibrated score of an input is
    the fraction of window scores below it, i.e. one minus its conformal p-value. Higher values
    mean more anomalous.

    The ACI update rule is:

    .. math::

        \\alpha_{t+1} = \\alpha_t + \\gamma (\\alpha - \\text{err}_t)

    where ``err_t = 1`` if ``score_t > qhat_t`` else ``0``, and ``qhat_t`` is the
    ``(1 - alpha_t)``-quantile of the recent scores. The result is clipped to ``[0, 1]`` when
    ``clip_alpha`` is enabled. A momentum variant replaces ``err_t`` with a weighted average of
    past errors. The momentum variant, fixed sliding score window, and conformal rank output are
    PySAD extensions beyond :cite:`gibbs2021adaptive`.

    As for model scores, higher calibrated values mean more anomalous. Alert when the calibrated
    score exceeds ``1 - alpha`` (approximately), or equivalently when the raw score exceeds
    :attr:`threshold_`.

    Args:
        alpha (float): Target miscoverage level in (0, 1). For example, ``alpha=0.1`` targets
            90% coverage. Lower values give wider intervals / higher thresholds.
        gamma (float): Step size for alpha adaptation. Larger values adapt faster to
            distribution shifts but increase volatility.
        window_size (int): Size of the sliding window for score quantiles. Must be at least 1
            (Default=500).
        method (str): Update method, either ``"simple"`` or ``"momentum"`` (Default="simple").
        momentum_bw (float): Bandwidth for momentum update, in (0, 1). Closer to 1 gives
            smoother, longer memory (Default=0.95).
        clip_alpha (bool): Whether to clip ``alpha_t`` to ``[0, 1]`` (Default=True).

    Raises:
        ValueError: If ``alpha`` not in (0, 1), ``gamma <= 0``, ``window_size < 1``,
            ``method`` not in {"simple", "momentum"}, or ``momentum_bw`` not in (0, 1).
        TypeError: If ``score`` is not convertible to float.

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

    _output_kind: ClassVar[Literal["score", "probability", "decision"]] = "probability"

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

    @property
    def alpha_t(self) -> float:
        return self._alphat

    @property
    def threshold_(self) -> float:
        return self._compute_qhat()

    def _validate_score(self, score: float) -> float:
        """Convert score to float, raising TypeError if not possible."""
        try:
            return float(score)
        except (TypeError, ValueError):
            raise TypeError(
                f"score must be convertible to float, got {type(score).__name__}: {score!r}"
            ) from None

    def _compute_qhat(self) -> float:
        """Compute the ``(1 - alpha_t)``-quantile of the non-missing scores in the window."""
        n = len(self._sorted)
        if n == 0:
            return 0.0
        q_level = float(np.clip(1.0 - self._alphat, 0.0, 1.0))
        idx = int(math.ceil(q_level * n)) - 1
        idx = max(0, min(idx, n - 1))
        return float(self._sorted[idx])

    def _miscoverage(self, score: float, qhat: float) -> float:
        """Return the ACI miscoverage event: 1 if ``score`` exceeds ``qhat``, else 0."""
        if self._alphat >= 1.0:
            return 1.0
        if self._alphat <= 0.0:
            return 0.0
        return float(score > qhat)

    def _update_alpha(self, err_t: float) -> float:
        """Update ``alpha_t`` using the ACI update rule and return the new value."""
        if self.method == "simple":
            self._alphat = self._alphat + self.gamma * (self.alpha - err_t)
        else:
            self._weighted_err_sum = self.momentum_bw * self._weighted_err_sum + err_t
            self._weight_sum = self.momentum_bw * self._weight_sum + 1.0
            weighted_err = self._weighted_err_sum / self._weight_sum
            self._alphat = self._alphat + self.gamma * (self.alpha - weighted_err)
        if self.clip_alpha:
            self._alphat = float(np.clip(self._alphat, 0.0, 1.0))
        return self._alphat

    def fit_partial(self, score: float) -> ConformalTSFMCalibrator:
        """Fit the next score: compute the miscoverage event, update ``alpha_t``, add the score.

        The quantile is computed from the window before the score is added, matching the online
        workflow where a score is evaluated against past data before the model learns from it.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        score = self._validate_score(score)
        qhat = self._compute_qhat()
        err_t = self._miscoverage(score, qhat)
        self._update_alpha(err_t)

        if len(self.window) == self.window_size:
            oldest = self.window._buffer.oldest()
            if oldest is not None and not math.isnan(oldest):
                del self._sorted[bisect_left(self._sorted, oldest)]
        self.window.update(score)
        if not math.isnan(score):
            insort(self._sorted, score)

        return self

    def transform_partial(self, score: float) -> float:
        """Transform the score using the current window, without changing the state.

        Returns the fraction of window scores below the input score, or 0.0 if the window is
        empty or the score is missing.

        Args:
            score (float): Input score.

        Returns:
            float: Fraction of window scores below the input score, or ``0.0`` if the window is empty.
        """
        n = len(self.window)
        s = self._validate_score(score)
        if n == 0 or math.isnan(s):
            return 0.0
        return bisect_left(self._sorted, float(s)) / n

    def fit_transform_partial(self, score: float) -> float:
        """Transform the score with the state before the score, then fit the score.

        This matches the standard online workflow: predict/transform using past data, then learn
        from the new observation.

        Args:
            score (float): Input score.

        Returns:
            float: Calibrated score (empirical p-value).
        """
        calibrated = self.transform_partial(score)
        self.fit_partial(score)
        return calibrated

    def transform(self, scores: NDArray[Any]) -> NDArray[Any]:
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

    def fit_transform(self, scores: NDArray[Any]) -> NDArray[Any]:
        """Fits and calibrates a 1-D score batch in ``O(N log W)`` time."""
        if isinstance(scores, np.ndarray) and scores.ndim == 1 and not np.isnan(scores).any():
            m = len(scores)
            out = np.empty(m, dtype=np.float64)
            if m == 0:
                return out
            buf = self.window._buffer
            sorted_vals = self._sorted
            w_size = self.window_size
            _bisect_left = bisect_left
            _insort = insort
            _isnan = math.isnan
            _miscoverage = self._miscoverage
            _update_alpha = self._update_alpha
            for i, s in enumerate(scores.tolist()):
                # Transform with the current state (before adding s).
                n_curr = len(buf)
                out[i] = 0.0 if n_curr == 0 else _bisect_left(sorted_vals, s) / n_curr

                # Fit: compute qhat from the current window, update alpha, then add s.
                qhat = self._compute_qhat()
                err_t = _miscoverage(s, qhat)
                _update_alpha(err_t)

                if len(buf) == w_size:
                    oldest = buf.oldest()
                    if oldest is not None and not _isnan(oldest):
                        del sorted_vals[_bisect_left(sorted_vals, oldest)]
                buf.append(s)
                _insort(sorted_vals, s)
            return out
        return super().fit_transform(scores)
