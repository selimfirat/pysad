from __future__ import annotations

import math
from collections import deque

import numpy as np

from pysad.core.base_postprocessor import BasePostprocessor


class AnomalyLikelihoodProbabilityCalibrator(BasePostprocessor):
    r"""Converts each score into the anomaly likelihood of HTM, with a rolling Gaussian of the short-term averages of the scores :cite:`ahmad2017unsupervised`.

    This calibrator ports the ``AnomalyLikelihood`` class of NuPIC (MIT license), which the "numenta" detector of the Numenta Anomaly Benchmark (NAB) uses. The short-term average :math:`\tilde{\mu}_t` is the mean of the last ``short_window_size`` scores, the current score included. The calibrator fits a Gaussian with mean :math:`\mu` and standard deviation :math:`\sigma` to the short-term averages of the last ``window_size`` scores, and returns the anomaly likelihood :math:`L_t = 1 - Q(|\tilde{\mu}_t - \mu| / \sigma)` (Eq. 4 & 5), where :math:`Q` is the Gaussian tail function. As for model scores, higher values mean more anomalous. The paper reports an anomaly when :math:`L_t \geq 1 - 10^{-5}`.

    The calibrator returns 0.5 for the first ``learning_period + num_estimation_samples`` scores. It then fits the Gaussian to the short-term averages of the scores after the first ``learning_period`` scores, and fits it again every ``reestimation_period`` scores. If the tail probability :math:`Q` of the last short-term average and of the current one are both at most :math:`10^{-5}`, the calibrator returns :math:`1 - 10^{-3}`, so that only the start of a run of anomalous scores reaches the threshold of the paper.

    The reference implementation differs from the paper in these ways, and the calibrator follows the reference implementation. The Gaussian is fitted to the short-term averages, not to the scores (Eq. 2 & 3), with the population variance. The mean is at least 0.03 and the variance is at least 0.0003, so these bounds assume scores between 0 and 1, as HTM gives. The tail is two-sided, so a short-term average far below the mean also gets a likelihood near 1. The default ``window_size`` is 8640, where the paper uses :math:`W = 8000` (Sec. 4). The paper has no flat 0.5 during the first ``learning_period + num_estimation_samples`` scores, and it updates the mean and variance at each step, not every ``reestimation_period`` scores. The paper also does not cap the likelihood at :math:`1 - 10^{-3}` after the first anomalous score of a run. :class:`GaussianTailProbabilityCalibrator` follows Eq. 2 to 6 of the paper instead.

    Differences from the reference implementation: The reference implementation also uses the input values of the detector, and returns a likelihood near 0.5 when their variance is below :math:`1.5 \times 10^{-5}`. A calibrator sees only the scores, so it omits this check. The tail function uses :math:`\sqrt{2}` instead of the approximation 1.4142.

    ``fit_transform_partial`` transforms each score with the state before the score, as in the reference implementation, and then fits the score.

    Args:
        learning_period (int): The number of first scores that the Gaussian ignores, while the detector learns. NAB uses half of its probationary period (Default=288).
        num_estimation_samples (int): The number of scores after ``learning_period`` that the first fit of the Gaussian uses (Default=100).
        window_size (int): The number of last scores that the Gaussian is fitted to. Must be at least ``num_estimation_samples``. The paper requires it to be much larger than ``short_window_size`` (Default=8640).
        reestimation_period (int): The number of scores between two fits of the Gaussian (Default=100).
        short_window_size (int): The number of last scores in the short-term average (Default=10).

    Raises:
        ValueError: If window_size is less than num_estimation_samples.
    """

    def __init__(
        self,
        learning_period: int = 288,
        num_estimation_samples: int = 100,
        window_size: int = 8640,
        reestimation_period: int = 100,
        short_window_size: int = 10,
    ) -> None:
        if window_size < num_estimation_samples:
            raise ValueError("window_size must be at least num_estimation_samples.")

        self.learning_period = learning_period
        self.num_estimation_samples = num_estimation_samples
        self.window_size = window_size
        self.reestimation_period = reestimation_period
        self.short_window_size = short_window_size

        self.scores_ = np.empty(window_size)
        self.short_window_: deque[float] = deque(maxlen=short_window_size)
        self.num_scores_ = 0
        self.mean_ = 0.0
        self.std_ = 0.0

    def fit_partial(self, score: float) -> AnomalyLikelihoodProbabilityCalibrator:
        """Fits particular (next) timestep's score to train the postprocessor.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.scores_[self.num_scores_ % self.window_size] = score
        self.short_window_.append(score)
        self.num_scores_ += 1

        num_init = self.learning_period + self.num_estimation_samples
        if self.num_scores_ == num_init or (
            self.num_scores_ > num_init and self.num_scores_ % self.reestimation_period == 0
        ):
            self._fit_gaussian()

        return self

    def transform_partial(self, score: float) -> float:
        """Transforms given score.

        Args:
            score (float): Input score.

        Returns:
            float: The anomaly likelihood of the score.
        """
        if self.num_scores_ < self.learning_period + self.num_estimation_samples:
            return 0.5

        total = sum(self.short_window_)
        prev_tail = self._tail(total / len(self.short_window_))
        if len(self.short_window_) == self.short_window_size:
            total -= self.short_window_[0]
            tail = self._tail((total + score) / self.short_window_size)
        else:
            tail = self._tail((total + score) / (len(self.short_window_) + 1))

        if tail <= 1e-5 and prev_tail <= 1e-5:
            tail = 1e-3

        return 1 - tail

    def fit_transform_partial(self, score: float) -> float:
        """Transforms the score with the state before the score, then fits the score.

        Args:
            score (float): Input score.

        Returns:
            float: The anomaly likelihood of the score.
        """
        likelihood = self.transform_partial(score)
        self.fit_partial(score)
        return likelihood

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Transforms a 1-D batch of scores against the current frozen state."""
        arr = np.asarray(scores, dtype=np.float64)
        if arr.ndim != 1:
            return super().transform(scores)
        n = len(arr)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        if self.num_scores_ < self.learning_period + self.num_estimation_samples:
            return np.full(n, 0.5, dtype=np.float64)
        total = sum(self.short_window_)
        prev_tail = self._tail(total / len(self.short_window_))
        if len(self.short_window_) == self.short_window_size:
            base_total = total - self.short_window_[0]
            denom = self.short_window_size
        else:
            base_total = total
            denom = len(self.short_window_) + 1
        avgs = (base_total + arr) / denom
        zs = np.abs(avgs - self.mean_) / (self.std_ * math.sqrt(2))
        _erfc = math.erfc
        tails = 0.5 * np.fromiter((_erfc(z) for z in zs.tolist()), dtype=np.float64, count=n)
        if prev_tail <= 1e-5:
            tails = np.where(tails <= 1e-5, 1e-3, tails)
        return 1.0 - tails

    def _fit_gaussian(self) -> None:
        start = self.num_scores_ % self.window_size
        if self.num_scores_ < self.window_size:
            scores = self.scores_[:start]
        else:
            scores = np.roll(self.scores_, -start)  # Oldest score first.
        # The scores that left the window count toward the learning period.
        skip = max(0, self.learning_period - max(0, self.num_scores_ - self.window_size))
        if len(scores) <= skip:
            self.mean_, self.std_ = 0.5, 1e3
            return

        # Short-term averages over the window, from its first score on: (len(scores),).
        sums = np.convolve(scores, np.ones(self.short_window_size))[: len(scores)]
        averages = sums[skip:] / np.minimum(
            np.arange(skip + 1, len(scores) + 1), self.short_window_size
        )
        self.mean_ = max(float(averages.mean()), 0.03)
        self.std_ = math.sqrt(max(float(averages.var()), 0.0003))

    def _tail(self, average: float) -> float:
        return 0.5 * math.erfc(abs(average - self.mean_) / (self.std_ * math.sqrt(2)))
