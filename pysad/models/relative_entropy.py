from scipy import stats
from pysad.core.base_model import BaseModel
import math
import numpy as np


class RelativeEntropy(BaseModel):
    """Relative entropy based anomaly detection model on univariate stream :cite:`ahmad2017unsupervised`. The implementation is based on `NAB-relative_entropy <https://github.com/numenta/NAB/blob/master/nab/detectors/relative_entropy/relative_entropy_detector.py>`_.

        Args:
            min_val (float): Minimum value of the univariate stream.
            max_val (float): Maximum value of the univariate stream.
            num_bins (int): Number of bins (Default=5).
            window_size (int): The size of the window (Default=52).
    """

    def __init__(self, min_val, max_val, num_bins=5, window_size=52):
        self.min_val = min_val
        self.max_val = max_val

        # Timeseries of the metric on which anomaly needs to be detected
        self.util = []

        # Number of bins into which util is to be quantized
        self.N_bins = num_bins

        # Window size
        self.W = window_size

        # Threshold against which the test statistic is compared. It is set to
        # the point in the chi-squared cdf with N-bins -1 degrees of freedom that
        #  corresponds to 0.99.
        self.T = stats.chi2.isf(0.01, self.N_bins - 1)

        # Tracks the current number of null hypothesis
        self.m = 0

        # Step size in time series quantization
        self.stepSize = (max_val - min_val) / self.N_bins

        # List of lists where P[i] indicates the empirical frequency of the ith
        # hypothesis.
        self.P = []

        # List where c[i] tracks the number of windows that agree with P[i]
        self.c = []

        self.P_hat = None

    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X (float): The instance to fit. Note that this model is univariate.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: Returns the self.
        """
        self.util.append(X)
        if len(self.util) >= self.W:

            # Extracting current window
            util_current = self.util[-self.W:]

            # Quantize window data points into discretized bin values
            B_current = [math.ceil((c - self.min_val) / self.stepSize) for c in
                         util_current]

            # Create a histogram of empirical frequencies for the current window
            # using B_current
            self.P_hat = np.histogram(B_current,
                                      bins=self.N_bins,
                                      range=(0, self.N_bins),
                                      density=True)[0]

            if self.m == 0:
                self.P.append(self.P_hat)
                self.c.append(1)
                self.m = 1

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance. Note that this method should be called after the fit_partial method.

        Args:
            X (any): (Ignored) The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        score = 0.0

        if len(self.util) >= self.W and self.m > 0 and self.P_hat is not None:
            score = self._get_aggreement_hypothesis(self.P_hat)

        return score

    def _get_aggreement_hypothesis(self, P_hat):
        """This function computes multinomial goodness-of-fit test. It calculates
        the relative entropy test statistic between P_hat and all `m` null
        hypothesis and compares it against the threshold `T` based on cdf of
        chi-squared distribution. The test relies on the observation that if the
        null hypothesis P is true, then as the number of samples grow the relative
        entropy converges to a chi-squared distribution1 with K-1 degrees of
        freedom.
        The function returns the index of hypothesis that agrees with minimum
        relative entropy. If all hypotheses disagree, the function returns -1.
        @param P_hat    (list)  Empirical frequencies of the current window.
        @return index   (int)   Index of the hypothesis with the minimum test
                                statistic.
        """

        index = -1
        minEntropy = float("inf")
        for i in range(self.m):
            entropy = 2 * self.W * stats.entropy(P_hat, self.P[i])
            if entropy < self.T and entropy < minEntropy:
                minEntropy = entropy
                index = i

        return index
