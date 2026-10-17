import math

import numpy as np

from pysad.core.base_postprocessor import BasePostprocessor
from pysad.statistics.average_meter import AverageMeter
from pysad.statistics.running_statistic import RunningStatistic
from pysad.statistics.variance_meter import VarianceMeter


class GaussianTailProbabilityCalibrator(BasePostprocessor):
    """Converts each score into a probability with the Gaussian tail function (Q-function) of the fitted scores :cite:`ahmad2017unsupervised`.

    The calibrator assumes that the scores follow a normal distribution. It transforms a score into the probability that a normal variable with the mean and variance of the fitted scores is lower than the score, i.e. one minus its Gaussian tail p-value. As for model scores, higher values mean more anomalous, e.g. alert when the calibrated score is above 0.95.

    Args:
        running_statistics (bool): Whether to calculate the mean and variance over a running window of ``window_size`` scores (Default=True).
        window_size (int): The size of the window for the running mean and variance. Ignored if ``running_statistics`` is False (Default=6400).
    """

    def __init__(self, running_statistics=True, window_size=6400):
        self.running_statistics = running_statistics
        self.window_size = window_size

        if self.running_statistics:
            self.avg_meter = RunningStatistic(AverageMeter, self.window_size)
            self.var_meter = RunningStatistic(VarianceMeter, self.window_size)
        else:
            self.avg_meter = AverageMeter()
            self.var_meter = VarianceMeter()

    def fit_partial(self, score):
        """Fits particular (next) timestep's score to train the postprocessor.

        Args:
            score (float): Input score.

        Returns:
            object: self.
        """
        self.avg_meter.update(score)
        self.var_meter.update(score)

        return self

    def transform_partial(self, score):
        """Transforms given score.

        Args:
            score (float): Input score.

        Returns:
            float: One minus the Gaussian tail p-value of the score.
        """
        mean = self.avg_meter.get()
        var = self.var_meter.get()
        if var > 0:
            std = np.sqrt(var)
        else:
            std = 1.0

        return 1 - self._qfunction(score, mean, std)

    def _qfunction(self, x, mean, std):
        """
        Given the normal distribution specified by the mean and standard deviation args, return the probability of getting samples > x. Implementation is adapted from the https://github.com/ish-vlad/Conformal-Anomaly-Detection/blob/22769b8d3cede7fabd978a36cdd2853255e450ac/scripts/nab_module/nab/detectors/gaussian/windowedGaussian_detector.py This is the
        Q-function: the tail probability of the normal distribution.
        """

        # Calculate the Q function with the complementary error function, explained
        # here:
        # http://www.gaussianwaves.com/2012/07/q-function-and-error-functions
        z = (x - mean) / std
        return 0.5 * math.erfc(z / math.sqrt(2))
