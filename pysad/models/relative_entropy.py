import numbers

from scipy import stats
from pysad.core.base_model import BaseModel
import numpy as np


def _is_positive_int(value):
    """Whether `value` is an integer >= 1: a Python or NumPy integer, but not a bool."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and value >= 1


class RelativeEntropy(BaseModel):
    """Relative entropy based anomaly detection model on univariate stream :cite:`wang2011statistical`, using the multinomial goodness-of-fit test with multiple null hypotheses (Fig. 1 of the paper), as evaluated in NAB :cite:`ahmad2017unsupervised`. The implementation is based on `NAB-relative_entropy <https://github.com/numenta/NAB/blob/master/nab/detectors/relative_entropy/relative_entropy_detector.py>`_. By default (`step=None`, resolved to `window_size`), windows are non-overlapping, as in the paper; pass `step=1` to reproduce NAB's sliding windows, which move by one value at a time. Each tested window's score goes to the value that closes it and every other value scores 0.0 (NAB tests a window at every value, and the paper flags windows rather than values), so with the default `step` only one value in every `window_size` can score nonzero. For the same reason, `score` on held-out values after `fit` scores each of them as the value following the fitted ones and returns all 0.0 unless that value would close a window (with the default `step`, unless the number of fitted values is one short of a multiple of `window_size`); pass `step=1` to fit and score separately. Unlike NAB, whose histogram puts the top two quantization levels in one bin, this implementation gives each of the `num_bins` equal-width buckets its own bin, as in the paper (Fig. 1, steps 3-4b), so its scores differ from NAB's. It follows NAB in scoring the first window 0.0, a case the paper is silent on. Following NAB, the anomaly score is 0.0 or 1.0: a window's histogram is compared against the learned hypotheses, and the score is 1.0 when the window agrees with no hypothesis, which is then added as a new hypothesis, and 0.0 otherwise. With NAB's rarity threshold `c_th` kept at 1, a window that agrees with an existing hypothesis always scores 0.0, since a hypothesis's count starts at 1 and is incremented before the comparison.

        Args:
            min_val (float): Minimum value of the univariate stream. Values below this are clipped to it.
            max_val (float): Maximum value of the univariate stream. Values above this are clipped to it.
            num_bins (int): Number of bins (Default=5).
            window_size (int): The size of the window (Default=52). Must be an int >= 1 (a NumPy integer is accepted), or `ValueError` is raised.
            step (int): Number of values between the ends of consecutive tested windows. `None` (default) resolves to `window_size`, giving the paper's non-overlapping windows; `step=1` reproduces NAB's sliding windows. Only the value that closes a tested window can score nonzero, so `step=1` is the setting for fitting and scoring separately. Must be `None` or an int >= 1 (a NumPy integer is accepted), or `ValueError` is raised.
    """

    def __init__(self, min_val, max_val, num_bins=5, window_size=52, step=None):
        if not _is_positive_int(window_size):
            raise ValueError("window_size must be an int >= 1.")
        if step is not None and not _is_positive_int(step):
            raise ValueError("step must be None or an int >= 1.")

        self.min_val = min_val
        self.max_val = max_val

        # Timeseries of the metric on which anomaly needs to be detected
        self.util = []

        # Number of bins into which util is to be quantized
        self.N_bins = num_bins

        # Window size, stored as a Python int even when given as a NumPy integer
        self.W = int(window_size)

        # Number of values between the ends of consecutive tested windows. None resolves to
        # W (the paper's non-overlapping windows); step=1 reproduces NAB's sliding windows.
        self.step = self.W if step is None else int(step)

        # Threshold against which the test statistic is compared. It is set to
        # the point in the chi-squared cdf with N-bins -1 degrees of freedom that
        #  corresponds to 0.99.
        self.T = stats.chi2.isf(0.01, self.N_bins - 1)

        # Tracks the current number of null hypothesis
        self.m = 0

        # Width of each quantization bucket (unrelated to `step`)
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
        """Fits the model to next instance: appends `X` to the window buffer and, when `X` closes a window (the window is full and ends `step` values after the previous tested window, counting from the first full window), either learns it as the first hypothesis or updates the agreeing hypothesis's count, adding it as a new hypothesis otherwise. Values that don't close a window are only appended to the window buffer; the learned hypotheses and counts are unchanged.

        Args:
            X (float): The instance to fit. Note that this model is univariate.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: Returns the self.

        Raises:
            ValueError: If `X` is NaN. The model is left unchanged.
        """
        x = self._value(X)
        self.util.append(x)

        if self.stepSize != 0.0 and self._closes_window(len(self.util)):
            P_hat, index = self._window_index(self.util[-self.W:])
            self._fit_window(P_hat, index)

        return self

    def score_partial(self, X):
        """Scores the window ending with the given instance, i.e., the last `W - 1` fitted values followed by `X`, if `X` would close a window. This method does not change the model, so `score` after `fit` scores every row as the value following the fitted ones: with the default `step`, it returns all 0.0 unless the number of fitted values is one short of a multiple of `window_size`. Use `step=1` to fit and score separately.

        Args:
            X (float): The instance to score. Note that this model is univariate.

        Returns:
            float: 1.0 if the window agrees with no hypothesis; with NAB's `c_th = 1`, a window that agrees with an existing hypothesis always scores 0.0. Also 0.0 before the window is full, when `X` doesn't close a window (per `step`, only the value that closes a window gets its score), before any hypothesis has been learned, or when `min_val == max_val` (bucket width `stepSize` is 0).

        Raises:
            ValueError: If `X` is NaN.
        """
        x = self._value(X)

        if self.stepSize == 0.0 or not self._closes_window(len(self.util) + 1) or self.m == 0:
            return 0.0

        start = max(0, len(self.util) - self.W + 1)
        window = self.util[start:] + [x]

        _, index = self._window_index(window)

        return self._score_window(index)

    def fit_score_partial(self, X, y=None):
        """Scores the window ending with the given instance and then fits the model to it, as NAB's detector does for each record. Only a value that closes a window (per `step`) gets that window's score and has the window fitted; any other value scores 0.0 and is only appended to the window buffer.

        Args:
            X (float): The instance to fit and score. Note that this model is univariate.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance, as in `score_partial`.

        Raises:
            ValueError: If `X` is NaN. The model is left unchanged.
        """
        x = self._value(X)
        self.util.append(x)

        if self.stepSize == 0.0 or not self._closes_window(len(self.util)):
            return 0.0

        # Computed once and shared: the score reads `index` before `_fit_window` changes
        # `self.P`/`self.c`/`self.m`, matching the score-then-fit order of `score_partial`
        # followed by `fit_partial`.
        P_hat, index = self._window_index(self.util[-self.W:])
        score = 0.0 if self.m == 0 else self._score_window(index)
        self._fit_window(P_hat, index)

        return score

    @staticmethod
    def _value(X):
        """Reads the single value of an instance, rejecting NaN before it can reach the window.

        Args:
            X (float): The instance. Note that this model is univariate.

        Returns:
            float: The value of `X` as a Python scalar.

        Raises:
            ValueError: If the value is NaN, which has no quantization level.
        """
        x = np.asarray(X).item()
        if np.isnan(x):
            raise ValueError("RelativeEntropy does not accept NaN values.")

        return x

    def _closes_window(self, length):
        """Whether the value bringing `util` to `length` closes a tested window: the window must
        be full and its end must land `step` values past the end of the previous tested window,
        counting from the first full window. With the default `step == W` this is true once per
        `W` values (the paper's non-overlapping windows); with `step == 1` it is true for every
        value from the first full window on (NAB's sliding windows).

        Args:
            length (int): The number of values in `util` counting the value that would close the
                window, i.e. `len(self.util)` in `fit_partial`/`fit_score_partial`, or
                `len(self.util) + 1` for the hypothetical window in `score_partial`.

        Returns:
            bool: True if the window ending at `length` should be fitted/scored.
        """
        return length >= self.W and (length - self.W) % self.step == 0

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
