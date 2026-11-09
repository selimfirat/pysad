from __future__ import annotations

import math
from collections import deque

from pysad.core.base_postprocessor import BasePostprocessor
from pysad.statistics.average_meter import AverageMeter
from pysad.statistics.running_statistic import RunningStatistic
from pysad.statistics.variance_meter import VarianceMeter


class GaussianTailProbabilityCalibrator(BasePostprocessor):
    r"""Converts each score into a probability with the Gaussian tail function (Q-function) of the fitted scores :cite:`ahmad2017unsupervised`.

    The calibrator assumes that the scores follow a normal distribution. It implements the anomaly likelihood of Ahmad et al. (2017) (Eq. 4 & 5) as :math:`L_t = 1 - Q((\tilde{\mu}_t - \mu_t) / \sigma_t)`. Here :math:`\mu_t` and :math:`\sigma_t` are the mean and standard deviation of the last ``window_size`` scores, and :math:`\tilde{\mu}_t` is the short-term average of the last ``short_window_size`` scores, where the score being transformed is the most recent one: it is averaged with the last ``short_window_size - 1`` fitted scores, so a score that was just fitted is counted once. Transforming a score never changes the state. ``fit_transform_partial`` fits the score first and then averages the short window as it is, so the streaming path also counts it once. The short-term average resists single-point noise spikes while responding to sustained score increases.

    When the fitted scores have zero variance, the calibrator returns the exact limit of the Gaussian tail as the standard deviation approaches zero: 1.0 if the short-term average is above the mean, 0.5 if it equals the mean, and 0.0 if it is below the mean.

    As for model scores, higher values mean more anomalous, e.g. alert when the calibrated score is above 0.95.

    Args:
        running_statistics (bool): Whether to calculate the mean and variance over a running window of ``window_size`` scores (Default=True).
        window_size (int): The size of the window for the running mean and variance. Ignored if ``running_statistics`` is False (Default=8000).
        short_window_size (int): The size of the short-term window for the short-term average :math:`\tilde{\mu}_t` (Default=10).

    Raises:
        ValueError: If window_size is less than 1 or short_window_size is less than 1.
    """

    def __init__(
        self,
        running_statistics: bool = True,
        window_size: int = 8000,
        short_window_size: int = 10,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        if short_window_size < 1:
            raise ValueError("short_window_size must be a positive integer.")

        self.running_statistics = running_statistics
        self.window_size = window_size
        self.short_window_size = short_window_size

        if self.running_statistics:
            self.avg_meter: RunningStatistic | AverageMeter = RunningStatistic(
                AverageMeter, self.window_size
            )
            self.var_meter: RunningStatistic | VarianceMeter = RunningStatistic(
                VarianceMeter, self.window_size
            )
        else:
            self.avg_meter = AverageMeter()
            self.var_meter = VarianceMeter()

        self.short_window: deque[float] = deque(maxlen=self.short_window_size)

    def fit_partial(self, score: float) -> GaussianTailProbabilityCalibrator:
        """Fits particular (next) timestep's score to train the postprocessor.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.avg_meter.update(score)
        self.var_meter.update(score)
        self.short_window.append(score)

        return self

    def transform_partial(self, score: float) -> float:
        """Transforms given score.

        The score is the most recent score in the short-term average: it is averaged with the
        last ``short_window_size - 1`` fitted scores (or used on its own when
        ``short_window_size`` is 1), so transforming a score does not change the state.

        Args:
            score (float): Input score.

        Returns:
            float: One minus the Gaussian tail p-value of the score, or 0.0 before the first fit.
        """
        if not self.short_window:
            return 0.0
        if self.short_window_size == 1:
            short_mean = score
        else:
            prev = list(self.short_window)[-(self.short_window_size - 1) :]
            short_mean = (sum(prev) + score) / (len(prev) + 1)

        return self._calibrate(short_mean)

    def fit_transform_partial(self, score: float) -> float:
        """Updates the postprocessor with the next score and transforms it.

        The score is counted once in the short-term average: it is fitted first, and the
        short-term average is then taken from the short window as it is.

        Args:
            score (float): Input score.

        Returns:
            float: One minus the Gaussian tail p-value of the score.
        """
        self.fit_partial(score)
        window = list(self.short_window)[-self.short_window_size :]

        return self._calibrate(sum(window) / len(window))

    def _calibrate(self, short_mean: float) -> float:
        mean = self.avg_meter.get()
        var = self.var_meter.get()
        if var <= 0:
            if short_mean > mean:
                return 1.0
            if short_mean < mean:
                return 0.0
            return 0.5

        std = math.sqrt(var)

        return 1 - self._qfunction(short_mean, mean, std)

    def _qfunction(self, x: float, mean: float, std: float) -> float:
        """
        Given the normal distribution specified by the mean and standard deviation args, return the probability of getting samples > x. Implementation is adapted from the https://github.com/ish-vlad/Conformal-Anomaly-Detection/blob/22769b8d3cede7fabd978a36cdd2853255e450ac/scripts/nab_module/nab/detectors/gaussian/windowedGaussian_detector.py This is the
        Q-function: the tail probability of the normal distribution.
        """

        # Calculate the Q function with the complementary error function, explained
        # here:
        # http://www.gaussianwaves.com/2012/07/q-function-and-error-functions
        z = (x - mean) / std
        return 0.5 * math.erfc(z / math.sqrt(2))
