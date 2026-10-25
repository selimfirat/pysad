from __future__ import annotations

import numpy as np

from pysad.core.base_transformer import BaseTransformer


class IdentityScaler(BaseTransformer):
    """Returns each feature vector unchanged, for pipelines that need a scaler slot."""

    def __init__(self) -> None:
        super().__init__(-1)

    def fit_partial(self, X: np.ndarray) -> IdentityScaler:
        """Convenience method that does not modify the input or the scaler.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            object: The scaler.
        """
        return self

    def transform_partial(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that does not modify the input.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (num_features,): The exact same input feature vector.
        """
        return X
