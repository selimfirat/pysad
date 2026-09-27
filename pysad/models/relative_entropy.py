from scipy import stats
from pysad.core.base_model import BaseModel
import math
import numpy as np


class RelativeEntropy(BaseModel):
    """Relative entropy based anomaly detection model on univariate stream :cite:`ahmad2017unsupervised`. The implementation is based on `NAB-relative_entropy <https://github.com/numenta/NAB/blob/master/nab/detectors/relative_entropy/relative_entropy_detector.py>`_. Following NAB, the anomaly score is 0.0 or 1.0: a window's histogram is compared against the learned hypotheses, and the score is 1.0 when the window agrees with no hypothesis (which is then added as a new hypothesis) or only with a hypothesis that is still rare, and 0.0 otherwise.

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

        # NAB's rarity threshold: a hypothesis counted at most this many times is still rare.
        self.c_th = 1

    def fit_partial(self, X, y=None):
        """Fits the model to next instance: appends `X` to the window and, once the window is full, either learns it as the first hypothesis or updates the agreeing hypothesis's count, adding it as a new hypothesis otherwise.

        Args:
            X (float): The instance to fit. Note that this model is univariate.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: Returns the self.
        """
        x = np.asarray(X).item()
        self.util.append(x)

        if self.stepSize != 0.0 and len(self.util) >= self.W:
            P_hat = self._histogram(self.util[-self.W:])

            if self.m == 0:
                self.P.append(P_hat)
                self.c.append(1)
                self.m = 1
            else:
                index = self._get_agreement_hypothesis(P_hat)
                if index != -1:
                    self.c[index] += 1
                else:
                    self.P.append(P_hat)
                    self.c.append(1)
                    self.m += 1

        return self

    def score_partial(self, X):
        """Scores the window ending with the given instance, i.e., the last `W - 1` fitted values followed by `X`. This method does not change the model.

        Args:
            X (float): The instance to score. Note that this model is univariate.

        Returns:
            float: 1.0 if the window agrees with no hypothesis or only with a still-rare one, 0.0 otherwise. Also 0.0 before the window is full or before any hypothesis has been learned.
        """
        x = np.asarray(X).item()

        if self.stepSize == 0.0:
            return 0.0

        start = max(0, len(self.util) - self.W + 1)
        window = self.util[start:] + [x]
        if len(window) < self.W or self.m == 0:
            return 0.0

        P_hat = self._histogram(window)
        index = self._get_agreement_hypothesis(P_hat)
        if index == -1:
            return 1.0

        return 1.0 if self.c[index] + 1 <= self.c_th else 0.0

    def fit_score_partial(self, X, y=None):
        """Scores the window ending with the given instance and then fits the model to it, as NAB's detector does for each record.

        Args:
            X (float): The instance to fit and score. Note that this model is univariate.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance, as in `score_partial`.
        """
        score = self.score_partial(X)
        self.fit_partial(X, y)

        return score

    def _histogram(self, window):
        """Computes the empirical frequency histogram `P_hat` of a window.

        Args:
            window (list of float): The values in the window, in order.

        Returns:
            np.float64 array of shape (N_bins,): The empirical frequencies of the quantized window.
        """
        B_current = [math.ceil((v - self.min_val) / self.stepSize) for v in window]

        return np.histogram(B_current, bins=self.N_bins, range=(0, self.N_bins), density=True)[0]

    def _get_agreement_hypothesis(self, P_hat):
        """This function computes multinomial goodness-of-fit test. It calculates
        the relative entropy test statistic between P_hat and all `m` null
        hypotheses and compares it against the threshold `T` based on cdf of
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
