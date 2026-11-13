from __future__ import annotations

import numpy as np

from pysad.core.base_transformer import BaseTransformer


def _unit_norm_rows(X: np.ndarray, pow: float) -> np.ndarray:
    """Divides each row of a 2-D batch by its norm.

    Args:
        X (np.float64 array of shape (num_instances, num_features)): Input feature batch.
        pow (float): The order of the norm.

    Returns:
        np.float64 array of shape (num_instances, num_features): Rows of unit norm, zeros where
        the row is all-zero, and non-finite values where the row holds them.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        # An over- or underflow here is detected below rather than reported to the caller.
        norms = np.linalg.norm(X, ord=pow, axis=1, keepdims=True)  # (num_instances, 1)
    nonzero = X.any(axis=1, keepdims=True)
    degenerate = (norms == 0.0) | np.isinf(norms)
    if degenerate.any():
        # Raising features to pow underflows a tiny row's norm to zero and overflows a huge row's
        # to infinity. Dividing such a row by its largest feature first keeps the norm in range,
        # and leaves the unit vector unchanged for every pow that scales with its input.
        rescale = degenerate & nonzero & np.isfinite(X).all(axis=1, keepdims=True)
        if rescale.any():
            X = X / np.where(rescale, np.abs(X).max(axis=1, keepdims=True), 1.0)
            norms = np.linalg.norm(X, ord=pow, axis=1, keepdims=True)
    # An all-zero row divides by one, keeping it zero instead of producing nan.
    np.copyto(norms, 1.0, where=~nonzero)

    return np.asarray(X / norms, dtype=np.float64)


class InstanceUnitNormScaler(BaseTransformer):
    """Scales each feature vector to unit norm.

    An all-zero vector has no direction and returns zeros instead of dividing by its zero norm.
    Vectors whose features are large or small enough to over- or underflow the norm are rescaled
    before it is taken, so they reach unit norm instead of zero or infinity.

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
            np.float64 array of shape (num_features,): Scaled feature vector, or zeros for an all-zero vector.
        """
        return _unit_norm_rows(X[None, :], self.pow)[0]

    def fit(self, X: np.ndarray) -> InstanceUnitNormScaler:
        """Returns the scaler unchanged without iterating the batch."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scales each row of a 2-D feature batch to unit norm, leaving all-zero rows at zero."""
        if isinstance(X, np.ndarray) and X.ndim == 2:
            if X.shape[0] == 0 or X.shape[1] == 0:
                return np.empty(X.shape, dtype=np.float64)
            return _unit_norm_rows(X, self.pow)
        return super().transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Scales each row of a 2-D feature batch to unit norm."""
        return self.transform(X)
