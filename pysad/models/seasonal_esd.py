import numpy as np
from scipy.stats import t

from pysad.core.base_model import BaseModel
from pysad.transform.preprocessing import ModifiedSTLResidualTransformer
from pysad.utils import Window


class SeasonalESD(BaseModel):
    """Window-based Seasonal ESD model :cite:`hochenbaum2017automatic`.

    This is the paper's S-ESD method: compute the modified STL residual
    ``X - seasonal - median(X)`` over a fixed-size PySAD window, then apply
    generalized ESD with mean and standard deviation to the residuals.

    Args:
        period (int): Number of observations in one seasonal period.
        window_size (int): Number of recent observations used for STL and ESD.
        max_anomalies (int): Maximum number of anomalies tested by ESD.
        alpha (float): ESD significance level. (Default=0.05).
        robust (bool): Whether to use robust STL fitting. (Default=True).
        **stl_kwargs: Additional keyword arguments passed to STL.
    """

    def __init__(
            self,
            period,
            window_size,
            max_anomalies,
            alpha=0.05,
            robust=True,
            **stl_kwargs):
        if max_anomalies < 1:
            raise ValueError("max_anomalies must be greater than 0.")

        if window_size < 2 * period:
            raise ValueError("window_size must be at least 2 * period.")

        if max_anomalies > int(window_size * 0.49):
            raise ValueError(
                "max_anomalies must be less than or equal to window_size * 0.49.")

        self.period = period
        self.window_size = window_size
        self.max_anomalies = max_anomalies
        self.alpha = alpha
        self.window = Window(window_size)
        self.residual_transformer = ModifiedSTLResidualTransformer(
            period=period,
            window_size=window_size,
            robust=robust,
            **stl_kwargs
        )

    def _as_value(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.shape != (1,):
            raise ValueError("SeasonalESD supports univariate inputs.")

        return X[0]

    def _center_scale(self, values):
        return np.mean(values), np.std(values, ddof=1)

    def _candidate_window(self, X):
        value = self._as_value(X)
        values = self.window.get() + [value]
        values = values[-self.window_size:]

        return np.asarray(values, dtype=np.float64)

    def _current_window(self):
        return np.asarray(self.window.get(), dtype=np.float64)

    def _critical_value(self, n, i):
        p = 1.0 - self.alpha / (2.0 * (n - i + 1))
        t_value = t.ppf(p, n - i - 1)
        return ((n - i) * t_value) / np.sqrt((n - i - 1 + t_value**2) * (n - i + 1))

    def _esd(self, values):
        remaining_values = np.asarray(values, dtype=np.float64)
        remaining_indices = np.arange(remaining_values.shape[0])
        candidates = []

        for i in range(1, self.max_anomalies + 1):
            center, scale = self._center_scale(remaining_values)
            if scale <= 1e-10:
                break

            deviations = np.abs(remaining_values - center) / scale
            local_idx = int(np.argmax(deviations))
            candidates.append((
                remaining_indices[local_idx],
                deviations[local_idx],
                self._critical_value(values.shape[0], i)
            ))
            remaining_values = np.delete(remaining_values, local_idx)
            remaining_indices = np.delete(remaining_indices, local_idx)

        selected_count = 0
        for i, (_, statistic, critical_value) in enumerate(candidates, start=1):
            if statistic > critical_value:
                selected_count = i

        return candidates[:selected_count]

    def fit_partial(self, X, y=None):
        """Adds the next instance to the model window."""
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
        """Scores whether the next instance is anomalous in a candidate window."""
        return self._score_window(self._candidate_window(X))

    def fit_score_partial(self, X, y=None):
        """Adds and scores the next instance without adding it twice."""
        self.fit_partial(X, y)
        return self._score_window(self._current_window())


class SeasonalHybridESD(SeasonalESD):
    """Window-based Seasonal Hybrid ESD model :cite:`hochenbaum2017automatic`.

    This is the paper's S-H-ESD method: use the same modified STL residual as
    :class:`SeasonalESD`, then replace ESD's mean and standard deviation with
    median and MAD-based scale in the test statistic.
    """

    def _center_scale(self, values):
        center = np.median(values)
        scale = 1.4826 * np.median(np.abs(values - center))
        return center, scale
