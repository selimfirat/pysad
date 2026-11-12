from __future__ import annotations

import numpy as np

from pysad.core.base_transformer import BaseTransformer


class InstanceStandardScaler(BaseTransformer):
    """Scale each feature vector to zero mean and unit standard deviation.

    Compute the mean and standard deviation across the features of each instance.
    Subtract the mean, then divide by the standard deviation.
    Constant finite vectors have zero variance and return zeros, including one-feature inputs.
    The scaler learns nothing from previous instances.
    """

    def __init__(self) -> None:
        super().__init__(-1)

    def fit_partial(self, X: np.ndarray) -> InstanceStandardScaler:
        """Return the scaler without learning from the input.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            InstanceStandardScaler: The scaler.
        """
        return self

    def transform_partial(self, X: np.ndarray) -> np.ndarray:
        """Scale one feature vector across its features.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (num_features,): Scaled features, or zeros for a constant finite vector.
        """
        if X.size and np.isfinite(X[0]) and (X == X[0]).all():
            return np.zeros_like(X, dtype=np.float64)

        X_mean = X.mean()
        X_std = X.std()

        return (X - X_mean) / X_std

    def fit(self, X: np.ndarray) -> InstanceStandardScaler:
        """Return the scaler without learning from the input."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale each row of a 2-D feature batch to zero mean and unit standard deviation."""
        if isinstance(X, np.ndarray) and X.ndim == 2:
            if X.shape[0] == 0 or X.shape[1] == 0:
                return np.empty(X.shape, dtype=np.float64)
            const_finite = np.isfinite(X[:, 0]) & (X == X[:, :1]).all(axis=1)
            if const_finite.all():
                return np.zeros(X.shape, dtype=np.float64)
            if not const_finite.any():
                means = X.mean(axis=1, keepdims=True)
                stds = X.std(axis=1, keepdims=True)
                return np.asarray((X - means) / stds, dtype=np.float64)
            out = np.zeros(X.shape, dtype=np.float64)
            sub = X[~const_finite]
            means = sub.mean(axis=1, keepdims=True)
            stds = sub.std(axis=1, keepdims=True)
            out[~const_finite] = (sub - means) / stds
            return out
        return super().transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Scale each row of a 2-D feature batch to zero mean and unit standard deviation."""
        return self.transform(X)
