from collections import deque

import numpy as np

from pysad.core.base_transformer import BaseTransformer


class SeasonalTrendDecomposer(BaseTransformer):
    """Streaming seasonal and trend decomposition preprocessor.

    The transformer subtracts a rolling mean trend estimate and a running
    average seasonal estimate for each position in the season. This is an
    online approximation intended for preprocessing before univariate anomaly
    scoring; it is not an STL implementation and does not reproduce the
    modified STL residual used by Seasonal ESD in
    :cite:`hochenbaum2017automatic`.

    Args:
        season_length (int): Number of observations in one seasonal period.
        trend_window (int): Number of recent observations used to estimate the
            trend. If None, defaults to ``season_length``.
    """

    def __init__(self, season_length, trend_window=None):
        super().__init__(-1)

        if season_length < 1:
            raise ValueError("season_length must be greater than 0.")

        if trend_window is None:
            trend_window = season_length
        elif trend_window < 1:
            raise ValueError("trend_window must be greater than 0.")

        self.season_length = season_length
        self.trend_window = trend_window
        self.window = deque(maxlen=trend_window)
        self.seasonal_sums = None
        self.seasonal_counts = None
        self.num_seen = 0

    def _as_array(self, X):
        if isinstance(X, tuple):
            X = X[0]

        return np.asarray(X)

    def _init_seasonal_state(self, X):
        self.seasonal_sums = np.zeros((self.season_length, X.shape[0]))
        self.seasonal_counts = np.zeros(self.season_length)

    def _trend(self):
        if len(self.window) == 0:
            return 0.0

        return np.mean(np.asarray(self.window), axis=0)

    def _seasonal(self, phase):
        if self.seasonal_counts[phase] == 0:
            return 0.0

        return self.seasonal_sums[phase] / self.seasonal_counts[phase]

    def fit_partial(self, X):
        """Fits the next timestep's values to train the decomposer.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            object: self.
        """
        X = self._as_array(X)

        if self.seasonal_sums is None:
            self._init_seasonal_state(X)

        phase = self.num_seen % self.season_length
        self.window.append(X)
        trend = self._trend()

        if len(self.window) == self.trend_window:
            self.seasonal_sums[phase] += X - trend
            self.seasonal_counts[phase] += 1

        self.num_seen += 1

        return self

    def transform_partial(self, X):
        """Transforms the next timestep by removing trend and seasonality.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            residual_X (np.float64 array of shape (num_features,)): Residual feature vector.
        """
        X = self._as_array(X)

        if self.seasonal_sums is None:
            self._init_seasonal_state(X)

        phase = self.num_seen % self.season_length
        return X - self._trend() - self._seasonal(phase)

    def fit_transform_partial(self, X):
        """Fits and transforms the next timestep.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            residual_X (np.float64 array of shape (num_features,)): Residual feature vector.
        """
        X = self._as_array(X)

        if self.seasonal_sums is None:
            self._init_seasonal_state(X)

        phase = self.num_seen % self.season_length
        self.window.append(X)
        trend = self._trend()

        if len(self.window) == self.trend_window:
            self.seasonal_sums[phase] += X - trend
            self.seasonal_counts[phase] += 1

        seasonal = self._seasonal(phase)
        self.num_seen += 1

        return X - trend - seasonal
