from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pysad.core.base_transformer import BaseTransformer


class IdentityScaler(BaseTransformer):
    """Returns each feature vector unchanged, for pipelines that need a scaler slot."""

    def __init__(self) -> None:
        super().__init__(-1)

    def fit_partial(self, X: NDArray[Any]) -> IdentityScaler:
        """Convenience method that does not modify the input or the scaler.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            object: The scaler.
        """
        return self

    def transform_partial(self, X: NDArray[Any]) -> NDArray[Any]:
        """Convenience method that does not modify the input.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (num_features,): The exact same input feature vector.
        """
        return X

    def fit(self, X: NDArray[Any]) -> IdentityScaler:
        """Returns the scaler unchanged without iterating the batch."""
        return self

    def transform(self, X: NDArray[Any]) -> NDArray[Any]:
        """Returns a float64 copy of a 2-D batch without per-instance iteration."""
        if isinstance(X, np.ndarray) and X.ndim == 2:
            return np.array(X, dtype=np.float64, copy=True)
        return super().transform(X)

    def fit_transform(self, X: NDArray[Any]) -> NDArray[Any]:
        """Returns a float64 copy of a 2-D batch without per-instance iteration."""
        return self.transform(X)
