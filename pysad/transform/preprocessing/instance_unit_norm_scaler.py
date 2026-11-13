from __future__ import annotations

import numpy as np

from pysad.core.base_transformer import BaseTransformer


class InstanceUnitNormScaler(BaseTransformer):
    """Scales each feature vector to unit norm.

    A vector with zero norm is returned unchanged.

    Args:
        pow (float): The order of the norm. ``pow=2`` gives the Euclidean norm (Default=2).
    """

    def __init__(self, pow: float = 2) -> None:
        super().__init__(-1)
        self.pow = pow

    def fit_partial(self, X: np.ndarray) -> InstanceUnitNormScaler:
        """Fits particular (next) timestep's features to train the scaler.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            object: self.
        """
        return self

    def transform_partial(self, X: np.ndarray) -> np.ndarray:
        """Scales particular (next) timestep's vector.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (num_features,): Scaled feature vector.
        """
        X_norm = np.linalg.norm(X, ord=self.pow)
        if X_norm == 0:
            return X.astype(np.float64)

        return X / X_norm

    def fit(self, X: np.ndarray) -> InstanceUnitNormScaler:
        """Returns the scaler unchanged without iterating the batch."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scales each row of a 2-D feature batch to unit norm."""
        if isinstance(X, np.ndarray) and X.ndim == 2:
            if X.shape[0] == 0 or X.shape[1] == 0:
                return np.empty(X.shape, dtype=np.float64)
            norms = np.linalg.norm(X, ord=self.pow, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return np.asarray(X / norms, dtype=np.float64)
        return super().transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Scales each row of a 2-D feature batch to unit norm."""
        return self.transform(X)
