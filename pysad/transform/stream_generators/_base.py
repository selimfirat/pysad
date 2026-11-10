from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_transformer import BaseTransformer


class _AdversarialStream(BaseTransformer):
    """Shared state for adversarial stream generators.

    A generator wraps a normal base stream and returns either each base point unchanged or an
    injected anomaly. The generator learns the normal distribution from the clean points passed
    to fit_partial, tracks a running per-feature mean and standard deviation, and exposes whether
    the last transformed point is an injected anomaly through is_anomaly.

    Args:
        random_state (int, np.random.RandomState or None): Seed or random number generator for
            the attack direction. None draws from NumPy's global random state, which
            pysad.utils.fix_seed seeds (Default=None).
    """

    def __init__(self, random_state: int | np.random.RandomState | None = None) -> None:
        super().__init__(0)
        self.random_state = random_state
        self._rng = check_random_state(random_state)
        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None
        self._count = 0
        self._direction: np.ndarray | None = None
        self._feature_count: int | None = None
        self._is_anomaly = False

    @property
    def is_anomaly(self) -> bool:
        """Whether the last transformed point is an injected anomaly."""
        return self._is_anomaly

    def _validate_point(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 1 or x.size == 0:
            raise ValueError(
                f"{type(self).__name__} expected a 1-D point, got an input of shape {np.shape(X)}."
            )
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{type(self).__name__} got a point with non-finite values.")
        if self._feature_count is not None and x.shape[0] != self._feature_count:
            raise ValueError(
                f"{type(self).__name__} got a point with {x.shape[0]} features after points with "
                f"{self._feature_count}."
            )
        self._feature_count = x.shape[0]
        return x

    def _ensure_direction(self, num_features: int) -> np.ndarray:
        if self._direction is None:
            direction = self._rng.standard_normal(num_features)
            norm = float(np.linalg.norm(direction))
            self._direction = direction / norm if norm > 0.0 else np.ones(num_features)
        assert self._direction is not None
        return self._direction

    def _update_normal(self, x: np.ndarray) -> None:
        self._ensure_direction(x.shape[0])
        if self._mean is None:
            self._mean = np.zeros(x.shape[0], dtype=np.float64)
            self._m2 = np.zeros(x.shape[0], dtype=np.float64)
        self._count += 1
        delta = x - self._mean
        self._mean = self._mean + delta / self._count
        self._m2 = self._m2 + delta * (x - self._mean)

    def _scale(self) -> np.ndarray:
        if self._count == 0 or self._mean is None or self._m2 is None:
            num_features = self._feature_count or 1
            return np.ones(num_features, dtype=np.float64)
        std = np.sqrt(self._m2 / self._count)
        return np.where(std > 0.0, std, 1.0)

    def fit_partial(self, X: np.ndarray) -> _AdversarialStream:
        """Learns the next clean point as part of the normal distribution.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            object: self.
        """
        x = self._validate_point(X)
        self._update_normal(x)
        self._is_anomaly = False
        return self
