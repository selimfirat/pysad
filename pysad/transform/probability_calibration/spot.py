from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.optimize import brentq

from pysad.core.base_postprocessor import BasePostprocessor


class SPOTProbabilityCalibrator(BasePostprocessor):
    r"""Converts each score into a probability with a Generalized Pareto tail that SPOT or DSPOT fits to the highest scores :cite:`siffer2017anomaly`.

    SPOT (Streaming Peaks-Over-Threshold) uses extreme value theory, so it makes no assumption about the distribution of the scores. It sets an initial threshold :math:`t` at the ``level`` quantile of the first ``num_init`` scores. The peaks are the scores above :math:`t`, minus :math:`t`. SPOT fits a Generalized Pareto distribution (GPD) with shape :math:`\gamma` and scale :math:`\sigma` to the peaks by maximum likelihood with Grimshaw's procedure (Section 3.4.2). The alarm threshold for the risk ``q`` is (Eq. 1)

    .. math::
        z_q = t + \frac{\sigma}{\gamma} \left( \left( \frac{q n}{N_t} \right)^{-\gamma} - 1 \right),

    where :math:`n` is the number of scores seen and :math:`N_t` is the number of peaks. A score above :math:`z_q` is an anomaly, and the calibrator does not learn from it (Algorithm 2). A score between :math:`t` and :math:`z_q` is a new peak, and the calibrator fits the GPD again.

    If ``depth`` is positive, the calibrator runs DSPOT (Algorithm 3) for streams with drift. DSPOT applies SPOT to the difference between each score and the mean of the last ``depth`` scores that are not anomalies. The first ``depth`` scores only fill this window.

    The paper defines thresholds, not probabilities. This calibrator returns one minus the tail probability :math:`\frac{N_t}{n} (1 + \gamma (x - t) / \sigma)^{-1/\gamma}` of the score :math:`x`, so a score at :math:`z_q` gets :math:`1 - q`, and higher values mean more anomalous. The tail model covers only the scores above :math:`t`, so the calibrator returns 0 for scores at or below :math:`t`, and for all scores before the initialization. ``threshold_`` gives :math:`z_q` on the scale of the raw scores, so you can also alert on ``score > calibrator.threshold_``.

    ``fit_transform_partial`` transforms each score with the state before the score, as in the paper, and then fits the score.

    Other deviations from the paper: The calibrator fits the GPD to the last ``max_peaks`` peaks, so that the memory and the cost of a fit stay bounded, but :math:`N_t` and :math:`n` count all peaks and scores. It finds the roots of Grimshaw's equation with a grid search and Brent's method instead of L-BFGS-B. If no initial score is above :math:`t`, :math:`z_q` is infinite until the first peak arrives.

    Args:
        q (float): The risk, i.e. the probability that a normal score is above the alarm threshold. Must be between 0 and 1 - ``level`` (Default=1e-4).
        num_init (int): The number of scores for the initialization (Default=1000).
        level (float): The quantile of the initial scores that sets the initial threshold :math:`t` (Default=0.98).
        depth (int): The number of recent scores for the moving average of DSPOT. 0 runs SPOT (Default=0).
        max_peaks (int): The maximum number of peaks that the GPD is fitted to (Default=1000).

    Raises:
        ValueError: If q is not between 0 and 1 - level.
    """

    def __init__(
        self,
        q: float = 1e-4,
        num_init: int = 1000,
        level: float = 0.98,
        depth: int = 0,
        max_peaks: int = 1000,
    ) -> None:
        if not 0 < q < 1 - level:
            raise ValueError("q must be between 0 and 1 - level.")

        self.q = q
        self.num_init = num_init
        self.level = level
        self.depth = depth
        self.max_peaks = max_peaks

        self.window_: deque[float] = deque(maxlen=depth)
        self.init_scores_: list[float] = []
        self.peaks_: deque[float] = deque(maxlen=max_peaks)
        self.num_peaks_ = 0
        self.num_scores_ = 0
        self.t_ = math.inf
        self.z_q_ = math.inf
        self.gamma_ = 0.0
        self.sigma_ = 0.0

    @property
    def threshold_(self) -> float:
        """float: The alarm threshold :math:`z_q` on the scale of the raw scores."""
        return self.z_q_ + self._baseline()

    def fit_partial(self, score: float) -> SPOTProbabilityCalibrator:
        """Fits particular (next) timestep's score to train the postprocessor.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        if len(self.window_) < self.depth:
            self.window_.append(score)
            return self

        x = score - self._baseline()
        if self.t_ == math.inf:
            self.init_scores_.append(x)
            if len(self.init_scores_) == self.num_init:
                self._initialize()
        elif x > self.z_q_:
            return self  # An anomaly does not update the calibrator.
        else:
            self.num_scores_ += 1
            if x > self.t_:
                self.peaks_.append(x - self.t_)
                self.num_peaks_ += 1
                self._fit()

        self.window_.append(score)
        return self

    def transform_partial(self, score: float) -> float:
        """Transforms given score.

        Args:
            score (float): Input score.

        Returns:
            float: One minus the tail probability of the score.
        """
        excess = score - self._baseline() - self.t_
        if not excess > 0:
            return 0.0
        if not self.num_peaks_:
            return 1.0

        if self.gamma_ == 0:
            tail = math.exp(-excess / self.sigma_)
        else:
            base = 1 + self.gamma_ * excess / self.sigma_
            tail = base ** (-1 / self.gamma_) if base > 0 else 0.0

        return 1 - self.num_peaks_ / self.num_scores_ * tail

    def fit_transform_partial(self, score: float) -> float:
        """Transforms the score with the state before the score, then fits the score.

        Args:
            score (float): Input score.

        Returns:
            float: One minus the tail probability of the score.
        """
        calibrated = self.transform_partial(score)
        self.fit_partial(score)
        return calibrated

    def _baseline(self) -> float:
        return sum(self.window_) / len(self.window_) if self.window_ else 0.0

    def _initialize(self) -> None:
        scores = np.array(self.init_scores_)
        self.init_scores_ = []
        self.t_ = float(np.quantile(scores, self.level))
        peaks = scores[scores > self.t_] - self.t_
        self.peaks_.extend(peaks.tolist())
        self.num_peaks_ = len(peaks)
        self.num_scores_ = len(scores)
        self._fit()

    def _fit(self) -> None:
        if not self.peaks_:
            return

        self.gamma_, self.sigma_ = _grimshaw(np.array(self.peaks_))
        ratio = self.q * self.num_scores_ / self.num_peaks_
        if self.gamma_ == 0:
            self.z_q_ = self.t_ - self.sigma_ * math.log(ratio)
        else:
            self.z_q_ = self.t_ + self.sigma_ / self.gamma_ * (ratio ** (-self.gamma_) - 1)


def _grimshaw(peaks: np.ndarray) -> tuple[float, float]:
    """Fits a GPD to positive peaks by maximum likelihood with Grimshaw's procedure.

    Returns:
        tuple[float, float]: The shape and the scale of the GPD.
    """
    mean, low, high = peaks.mean(), peaks.min(), peaks.max()

    def w(x: np.ndarray | float) -> np.ndarray:
        xy = np.multiply.outer(x, peaks)  # (num_x, num_peaks)
        return (1 / (1 + xy)).mean(axis=-1) * (1 + np.log1p(xy).mean(axis=-1)) - 1

    # The roots of w other than 0 are in (-1 / high, 0) and (0, 2 (mean - low) / low^2].
    # The grids are dense near both ends of each interval.
    s = np.r_[np.geomspace(1e-6, 0.5, 20), 1 - np.geomspace(0.5, 1e-6, 20)[1:]]  # (39,) in (0, 1)
    grids = [-s / high]
    upper = 2 * (mean - low) / low**2
    if upper > 1e-6 / high:
        grids.append(np.geomspace(1e-6 / high, upper, 40))

    candidates = [(0.0, float(mean))]  # gamma = 0 is the exponential distribution.
    for grid in grids:
        values = w(grid)
        for i in np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:])):
            x = brentq(w, grid[i], grid[i + 1])
            gamma = float(np.log1p(x * peaks).mean())
            candidates.append((gamma, gamma / x))

    return max(candidates, key=lambda c: _log_likelihood(peaks, *c))


def _log_likelihood(peaks: np.ndarray, gamma: float, sigma: float) -> float:
    if gamma == 0:
        return float(-len(peaks) * math.log(sigma) - peaks.sum() / sigma)
    return float(
        -len(peaks) * math.log(sigma) - (1 + 1 / gamma) * np.log1p(gamma / sigma * peaks).sum()
    )
