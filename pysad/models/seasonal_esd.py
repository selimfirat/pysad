from functools import lru_cache

import numpy as np
from scipy.stats import t

from pysad.core.base_model import BaseModel
from pysad.transform.preprocessing import ModifiedSTLResidualTransformer
from pysad.utils import Window
from pysad.utils.validation import check_univariate


@lru_cache(maxsize=1024)
def _critical_value(alpha, num_tails, n, i):
    # The critical value of the i-th ESD test on n values. It depends only on these numbers, so it
    # is computed once instead of once per scored instance.
    p = 1.0 - alpha / (num_tails * (n - i + 1))
    t_value = t.ppf(p, n - i - 1)
    return ((n - i) * t_value) / np.sqrt((n - i - 1 + t_value**2) * (n - i + 1))


class SeasonalESD(BaseModel):
    """Flags seasonal anomalies in a univariate stream with the Seasonal ESD (S-ESD) test over a sliding window :cite:`hochenbaum2017automatic`.

    This is the S-ESD method of the paper (Eq. 5). It computes the modified STL residual ``X - seasonal - median(X)`` over a fixed-size window, with the periodic STL of R (see :class:`pysad.transform.preprocessing.ModifiedSTLResidualTransformer`), then applies generalized ESD with the mean and the standard deviation to the residuals. The reference implementation of the paper, Twitter's AnomalyDetection R package, has no S-ESD: it always uses the median and the MAD, as :class:`SeasonalHybridESD` does. This model follows the R package in everything else. The test stops when the scale of the remaining values is at most 1e-10, where R stops only when the MAD is exactly 0, so that the floating-point noise of a series without anomalies raises no alerts. The paper runs the test on a full batch. This model runs it on the most recent ``window_size`` values for each instance.

    The score of an instance is the ESD test statistic of the newest value if the test flags it as an anomaly, else 0.0. The score is 0.0 until the window holds ``window_size`` values.

    Args:
        period (int): Number of observations in one seasonal period. It must be at least 2.
        window_size (int): Number of recent observations used for STL and ESD. It must be greater than ``2 * period``.
        max_anomalies (float): Maximum number of anomalies that ESD tests in each window, as a fraction of ``window_size``, greater than 0 and at most 0.49. The test looks for ``max(1, int(max_anomalies * window_size))`` anomalies (Default=0.1).
        alpha (float): ESD significance level, strictly between 0 and 1 (Default=0.05).
        direction (str): Which anomalies the test flags: ``"pos"`` for high values, ``"neg"`` for low values, or ``"both"`` for both, with the critical values of a two-sided test (Default="pos").
        robust (bool): Whether to use robust STL fitting (Default=True).

    Raises:
        ValueError: If ``period`` is less than 2, ``window_size`` is at most ``2 * period``, ``max_anomalies`` is not greater than 0 and at most 0.49, ``alpha`` is not strictly between 0 and 1, or ``direction`` is not ``"pos"``, ``"neg"`` or ``"both"``.
    """

    def __init__(
        self, period, window_size, max_anomalies=0.1, alpha=0.05, direction="pos", robust=True
    ):
        if window_size <= 2 * period:
            raise ValueError("window_size must be greater than 2 * period.")

        if not 0 < max_anomalies <= 0.49:
            raise ValueError("max_anomalies must be greater than 0 and at most 0.49.")

        if not np.isfinite(alpha) or not (0 < alpha < 1):
            raise ValueError("alpha must be finite and between 0 and 1.")

        if direction not in ("pos", "neg", "both"):
            raise ValueError("direction must be 'pos', 'neg' or 'both'.")

        self.period = period
        self.window_size = window_size
        self.max_anomalies = max_anomalies
        self.alpha = alpha
        self.direction = direction
        self.window = Window(window_size)
        self.residual_transformer = ModifiedSTLResidualTransformer(
            period=period, window_size=window_size, robust=robust
        )

    def _as_value(self, X):
        return check_univariate(X, "SeasonalESD")

    def _center_scale(self, values):
        return np.mean(values), np.std(values, ddof=1)

    def _candidate_window(self, X):
        value = self._as_value(X)
        values = self.window.get() + [value]
        values = values[-self.window_size :]

        return np.asarray(values, dtype=np.float64)

    def _current_window(self):
        return np.asarray(self.window.get(), dtype=np.float64)

    def _critical_value(self, n, i):
        return _critical_value(self.alpha, 2.0 if self.direction == "both" else 1.0, n, i)

    def _esd(self, values):
        remaining_values = np.asarray(values, dtype=np.float64)
        remaining_indices = np.arange(remaining_values.shape[0])
        candidates = []

        for i in range(1, max(1, int(self.max_anomalies * values.shape[0])) + 1):
            center, scale = self._center_scale(remaining_values)
            if scale <= 1e-10:
                break

            deviations = remaining_values - center
            if self.direction == "neg":
                deviations = -deviations
            elif self.direction == "both":
                deviations = np.abs(deviations)
            deviations /= scale
            local_idx = int(np.argmax(deviations))
            candidates.append(
                (
                    remaining_indices[local_idx],
                    deviations[local_idx],
                    self._critical_value(values.shape[0], i),
                )
            )
            remaining_values = np.delete(remaining_values, local_idx)
            remaining_indices = np.delete(remaining_indices, local_idx)

        selected_count = 0
        for i, (_, statistic, critical_value) in enumerate(candidates, start=1):
            if statistic > critical_value:
                selected_count = i

        return candidates[:selected_count]

    def fit_partial(self, X, y=None):
        """Adds the next instance to the model window.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.
            y (int or None): Ignored. Present for API consistency (Default=None).

        Returns:
            object: self.
        """
        self.window.update(self._as_value(X))
        return self

    def _score_window(self, values):
        if values.shape[0] < self.window_size:
            return 0.0

        residuals = self.residual_transformer.transform_window(values)
        anomalies = self._esd(residuals)
        latest_idx = values.shape[0] - 1

        for idx, statistic, _ in anomalies:
            if idx == latest_idx:
                return statistic

        return 0.0

    def score_partial(self, X):
        """Scores the next instance in the window that ends with it, without adding it to the window.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.

        Returns:
            float: The ESD test statistic of ``X`` if the test flags it as an anomaly, else 0.0.
        """
        return self._score_window(self._candidate_window(X))

    def fit_score_partial(self, X, y=None):
        """Adds the next instance to the window and scores it.

        Args:
            X (np.float64 array of shape (1,)): The next univariate value.
            y (int or None): Ignored. Present for API consistency (Default=None).

        Returns:
            float: The ESD test statistic of ``X`` if the test flags it as an anomaly, else 0.0.
        """
        self.fit_partial(X, y)
        return self._score_window(self._current_window())


class SeasonalHybridESD(SeasonalESD):
    """Flags seasonal anomalies in a univariate stream with the Seasonal Hybrid ESD (S-H-ESD) test over a sliding window :cite:`hochenbaum2017automatic`.

    This is the S-H-ESD method of the paper. It uses the same periodic modified STL residual as :class:`SeasonalESD`, then replaces the mean and the standard deviation of ESD with the median and the MAD-based scale (1.4826 times the MAD) in the test statistic. For each window, the test flags the anomalies that ``AnomalyDetectionVec(window, max_anoms = max_anomalies, direction = direction, alpha = alpha, period = period)`` of Twitter's AnomalyDetection R package, the reference implementation of the paper, flags. The exception is an even ``window_size``, for which the robust STL sometimes differs from R, as :class:`pysad.transform.preprocessing.ModifiedSTLResidualTransformer` documents. Use an odd ``window_size`` to match R. The test stops when the scale of the remaining values is at most 1e-10, where R stops only when the MAD is exactly 0, so that the floating-point noise of a series without anomalies raises no alerts. The paper runs the test on a full batch. This model runs it on the most recent ``window_size`` values for each instance.

    The score of an instance is the test statistic of the newest value if the test flags it as an anomaly, else 0.0. The score is 0.0 until the window holds ``window_size`` values.

    Args:
        period (int): Number of observations in one seasonal period. It must be at least 2.
        window_size (int): Number of recent observations used for STL and ESD. It must be greater than ``2 * period``.
        max_anomalies (float): Maximum number of anomalies that ESD tests in each window, as a fraction of ``window_size``, greater than 0 and at most 0.49. The test looks for ``max(1, int(max_anomalies * window_size))`` anomalies (Default=0.1).
        alpha (float): ESD significance level, strictly between 0 and 1 (Default=0.05).
        direction (str): Which anomalies the test flags: ``"pos"`` for high values, ``"neg"`` for low values, or ``"both"`` for both, with the critical values of a two-sided test (Default="pos").
        robust (bool): Whether to use robust STL fitting (Default=True).

    Raises:
        ValueError: If ``period`` is less than 2, ``window_size`` is at most ``2 * period``, ``max_anomalies`` is not greater than 0 and at most 0.49, ``alpha`` is not strictly between 0 and 1, or ``direction`` is not ``"pos"``, ``"neg"`` or ``"both"``.
    """

    def _center_scale(self, values):
        center = np.median(values)
        scale = 1.4826 * np.median(np.abs(values - center))
        return center, scale
