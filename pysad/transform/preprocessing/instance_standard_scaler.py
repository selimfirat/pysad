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
