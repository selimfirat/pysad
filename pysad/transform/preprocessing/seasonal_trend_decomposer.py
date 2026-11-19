from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import ArrayLike

from pysad.core.base_transformer import BaseTransformer


class SeasonalTrendDecomposer(BaseTransformer):
    """Removes a rolling trend and a running seasonal estimate from each feature vector.

    The transformer subtracts a rolling mean trend estimate and a running
    average seasonal estimate for each position in the season. This is an
    online approximation intended for preprocessing before univariate anomaly
    scoring; it is not an STL implementation and does not reproduce the
    modified STL residual used by Seasonal ESD in
    :cite:`hochenbaum2017automatic`.

    Args:
        season_length (int): Number of observations in one seasonal period.
        trend_window (int or None): Number of recent observations used to estimate the trend. If None, the decomposer uses ``season_length`` (Default=None).
    """

    def __init__(self, season_length: int, trend_window: int | None = None) -> None:
        super().__init__(-1)

        if season_length < 1:
            raise ValueError("season_length must be greater than 0.")

        if trend_window is None:
            trend_window = season_length
        elif trend_window < 1:
            raise ValueError("trend_window must be greater than 0.")

        self.season_length = season_length
        self.trend_window = trend_window
        self.window: deque[np.ndarray] = deque()
        self._window_sum: np.ndarray | None = None
        self.seasonal_sums: np.ndarray | None = None
        self.seasonal_counts: np.ndarray | None = None
        self.num_seen = 0

    def _as_array(self, X: ArrayLike) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)

    def _init_seasonal_state(self, X: np.ndarray) -> None:
        self.seasonal_sums = np.zeros((self.season_length, X.shape[0]), dtype=np.float64)
        self.seasonal_counts = np.zeros(self.season_length, dtype=np.float64)
        self._window_sum = np.zeros(X.shape[0], dtype=np.float64)

    def _push_window(self, X: np.ndarray) -> np.ndarray:
        assert self._window_sum is not None
        if len(self.window) == self.trend_window:
            evicted = self.window.popleft()
            self.window.append(X)
            if (self.num_seen & 4095) == 4095:
                self._window_sum = np.sum(np.asarray(self.window), axis=0)
            else:
                self._window_sum = self._window_sum - evicted + X
        else:
            self.window.append(X)
            self._window_sum = self._window_sum + X
        return self._window_sum / len(self.window)

    def _trend(self) -> float | np.ndarray:
        if not self.window or self._window_sum is None:
            return 0.0

        return self._window_sum / len(self.window)

    def _seasonal(self, phase: int) -> float | np.ndarray:
        if (
            self.seasonal_counts is None
            or self.seasonal_sums is None
            or self.seasonal_counts[phase] == 0
        ):
            return 0.0

        return self.seasonal_sums[phase] / self.seasonal_counts[phase]

    def fit_partial(self, X: np.ndarray) -> SeasonalTrendDecomposer:
        """Fits the next timestep's values to train the decomposer.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            object: self.
        """
        X = self._as_array(X)

        if self.seasonal_sums is None:
            self._init_seasonal_state(X)
        assert self.seasonal_sums is not None
        assert self.seasonal_counts is not None

        phase = self.num_seen % self.season_length
        trend = self._push_window(X)

        if len(self.window) == self.trend_window:
            self.seasonal_sums[phase] += X - trend
            self.seasonal_counts[phase] += 1

        self.num_seen += 1

        return self

    def transform_partial(self, X: np.ndarray) -> np.ndarray:
        """Transforms the next timestep by removing trend and seasonality.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (num_features,): Residual feature vector.
        """
        X = self._as_array(X)

        if self.seasonal_sums is None:
            self._init_seasonal_state(X)

        phase = self.num_seen % self.season_length
        return X - self._trend() - self._seasonal(phase)

    def fit_transform_partial(self, X: np.ndarray) -> np.ndarray:
        """Fits and transforms the next timestep.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (num_features,): Residual feature vector.
        """
        X = self._as_array(X)

        if self.seasonal_sums is None:
            self._init_seasonal_state(X)
        assert self.seasonal_sums is not None
        assert self.seasonal_counts is not None

        phase = self.num_seen % self.season_length
        trend = self._push_window(X)

        if len(self.window) == self.trend_window:
            self.seasonal_sums[phase] += X - trend
            self.seasonal_counts[phase] += 1

        seasonal = self._seasonal(phase)
        self.num_seen += 1

        return X - trend - seasonal

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits and transforms all instances in order."""
        if self.num_seen > 0 or not isinstance(X, np.ndarray) or X.ndim != 2 or len(X) == 0:
            return super().fit_transform(X)

        Xf = np.asarray(X, dtype=np.float64)
        n, d = Xf.shape
        w = self.trend_window
        s = self.season_length

        self.seasonal_sums = np.zeros((s, d), dtype=np.float64)
        self.seasonal_counts = np.zeros(s, dtype=np.float64)

        csum = np.empty((n + 1, d), dtype=np.float64)
        csum[0] = 0.0
        np.cumsum(Xf, axis=0, out=csum[1:])

        trend = np.empty((n, d), dtype=np.float64)
        warmup = min(n, w - 1)
        if warmup > 0:
            trend[:warmup] = (
                csum[1 : warmup + 1] / np.arange(1, warmup + 1, dtype=np.float64)[:, None]
            )
        if n >= w:
            trend[w - 1 :] = (csum[w:] - csum[: n - w + 1]) / w

        seasonal = np.zeros((n, d), dtype=np.float64)
        if n >= w:
            detrended = Xf[w - 1 :] - trend[w - 1 :]
            m = len(detrended)
            for r in range(min(s, m)):
                phase = (w - 1 + r) % s
                phase_csum = np.cumsum(detrended[r::s], axis=0)
                counts = np.arange(1, len(phase_csum) + 1, dtype=np.float64)
                seasonal[w - 1 + r :: s] = phase_csum / counts[:, None]
                self.seasonal_sums[phase] = phase_csum[-1]
                self.seasonal_counts[phase] = float(len(phase_csum))

        tail = Xf[max(0, n - w) :]
        self.window = deque(tail[i].copy() for i in range(len(tail)))
        self._window_sum = tail.sum(axis=0)
        self.num_seen = n

        return Xf - trend - seasonal
