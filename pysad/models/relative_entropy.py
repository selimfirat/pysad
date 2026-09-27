from scipy import stats
from pysad.core.base_model import BaseModel
import numpy as np


class RelativeEntropy(BaseModel):
    """Relative entropy based anomaly detection model on univariate stream :cite:`wang2011statistical`, using the multinomial goodness-of-fit test with multiple null hypotheses (Fig. 1 of the paper), as evaluated in NAB :cite:`ahmad2017unsupervised`. The implementation is based on `NAB-relative_entropy <https://github.com/numenta/NAB/blob/master/nab/detectors/relative_entropy/relative_entropy_detector.py>`_: it differs from the paper in that windows slide one value at a time instead of being non-overlapping. Unlike NAB, whose histogram puts the top two quantization levels in one bin, this implementation gives each of the `num_bins` equal-width buckets its own bin, as in the paper (Fig. 1, steps 3-4b), so its scores differ from NAB's. It follows NAB in scoring the first window 0.0, a case the paper is silent on. Following NAB, the anomaly score is 0.0 or 1.0: a window's histogram is compared against the learned hypotheses, and the score is 1.0 when the window agrees with no hypothesis, which is then added as a new hypothesis, and 0.0 otherwise. With NAB's rarity threshold `c_th` kept at 1, a window that agrees with an existing hypothesis always scores 0.0, since a hypothesis's count starts at 1 and is incremented before the comparison.

        Args:
            min_val (float): Minimum value of the univariate stream. Values below this are clipped to it.
            max_val (float): Maximum value of the univariate stream. Values above this are clipped to it.
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

        # Rows are the empirical frequencies of the learned hypotheses. Grows by one row
        # whenever a new hypothesis is added.
        self.P = np.empty((0, self.N_bins))

        # List where c[i] tracks the number of windows that agree with P[i]
        self.c = []

        # NAB's rarity threshold, kept at 1. With c_th = 1 no accepted hypothesis is rare:
        # counts start at 1 and are incremented before the comparison.
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
            P_hat, index = self._window_index(self.util[-self.W:])
            self._fit_window(P_hat, index)

        return self

    def score_partial(self, X):
        """Scores the window ending with the given instance, i.e., the last `W - 1` fitted values followed by `X`. This method does not change the model.

        Args:
            X (float): The instance to score. Note that this model is univariate.

        Returns:
            float: 1.0 if the window agrees with no hypothesis; with NAB's `c_th = 1`, a window that agrees with an existing hypothesis always scores 0.0. Also 0.0 before the window is full, before any hypothesis has been learned, or when `min_val == max_val` (step size 0).
        """
        x = np.asarray(X).item()

        if self.stepSize == 0.0:
            return 0.0

        start = max(0, len(self.util) - self.W + 1)
        window = self.util[start:] + [x]
        if len(window) < self.W or self.m == 0:
            return 0.0

        _, index = self._window_index(window)

        return self._score_window(index)

    def fit_score_partial(self, X, y=None):
        """Scores the window ending with the given instance and then fits the model to it, as NAB's detector does for each record.

        Args:
            X (float): The instance to fit and score. Note that this model is univariate.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance, as in `score_partial`.
        """
        x = np.asarray(X).item()
        self.util.append(x)

        if self.stepSize == 0.0 or len(self.util) < self.W:
            return 0.0

        # Computed once and shared: the score reads `index` before `_fit_window` changes
        # `self.P`/`self.c`/`self.m`, matching the score-then-fit order of `score_partial`
        # followed by `fit_partial`.
        P_hat, index = self._window_index(self.util[-self.W:])
        score = 0.0 if self.m == 0 else self._score_window(index)
        self._fit_window(P_hat, index)

        return score

    def _window_index(self, window):
        """Computes a window's empirical histogram and the index of the hypothesis it agrees with.

        Args:
            window (list of float): The values in the window, in order.

        Returns:
            tuple: `(P_hat, index)`, where `P_hat` is the np.float64 array of shape `(N_bins,)`
                returned by `_histogram` and `index` is the return value of
                `_get_agreement_hypothesis(P_hat)`.
        """
        P_hat = self._histogram(window)
        index = self._get_agreement_hypothesis(P_hat)

        return P_hat, index

    def _score_window(self, index):
        """Scores a window from the index of the hypothesis it agrees with.

        Args:
            index (int): The return value of `_get_agreement_hypothesis` for the window.

        Returns:
            float: 1.0 if the window agrees with no hypothesis; with NAB's `c_th = 1`, a window that agrees with an existing hypothesis always scores 0.0.
        """
        if index == -1:
            return 1.0

        return 1.0 if self.c[index] + 1 <= self.c_th else 0.0

    def _fit_window(self, P_hat, index):
        """Fits a window from its histogram and the index of the hypothesis it agrees with: updates
        the agreeing hypothesis's count, or adds `P_hat` as a new hypothesis if none agrees.

        Args:
            P_hat (np.float64 array of shape (N_bins,)): The empirical frequencies of the window,
                used only when a new hypothesis needs to be added.
            index (int): The return value of `_get_agreement_hypothesis` for the window.
        """
        if index != -1:
            self.c[index] += 1
        else:
            self._add_hypothesis(P_hat)

    def _add_hypothesis(self, P_hat):
        """Learns `P_hat` as a new hypothesis with a window count of 1.

        Args:
            P_hat (np.float64 array of shape (N_bins,)): The empirical frequencies of the window.
        """
        self.P = np.vstack([self.P, P_hat])
        self.c.append(1)
        self.m += 1

    def _histogram(self, window):
        """Computes the empirical frequency histogram `P_hat` of a window.

        Args:
            window (list of float): The values in the window, in order.

        Returns:
            np.float64 array of shape (N_bins,): The empirical frequencies of the quantized window.
        """
        values = np.clip(np.asarray(window, dtype=np.float64), self.min_val, self.max_val)
        # Levels 1 to N_bins map to bins 0 to N_bins - 1, one bucket per level, as in the
        # paper's quantizer (Fig. 1, steps 3-4b). Clipping the level to [1, N_bins] puts
        # min_val in the first bucket (ceil((min_val - min_val) / stepSize) == 0 otherwise)
        # and also absorbs the floating-point round-off that can put a value already clipped
        # to max_val one level past N_bins (e.g. ceil((100-0)/(100/29)) == 30).
        B_current = np.clip(np.ceil((values - self.min_val) / self.stepSize), 1, self.N_bins).astype(int)

        return np.bincount(B_current - 1, minlength=self.N_bins) / len(window)

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
        @param P_hat    (np.float64 array of shape (N_bins,))  Empirical frequencies of the current window.
        @return index   (int)   Index of the hypothesis with the minimum test
                                statistic.
        """
        if self.m == 0:
            return -1

        # Relative entropy of P_hat against every learned hypothesis in one call, instead of
        # looping in Python and calling scipy.stats.entropy once per hypothesis. P_hat is
        # broadcast to self.P's shape explicitly because scipy < 1.12 normalizes `pk` along
        # `axis` before broadcasting, which raises AxisError for a 1-D `pk` with axis=1.
        entropies = 2 * self.W * stats.entropy(np.broadcast_to(P_hat, self.P.shape), self.P, axis=1)
        candidates = np.flatnonzero(entropies < self.T)
        if candidates.size == 0:
            return -1

        return int(candidates[np.argmin(entropies[candidates])])
