from __future__ import annotations

import numpy as np

from pysad.core.base_transformer import BaseTransformer


def _standardize_rows(X: np.ndarray) -> np.ndarray:
    """Centre each row of a 2-D batch and divide it by its standard deviation.

    Args:
        X (np.float64 array of shape (num_instances, num_features)): Input feature batch, holding
            no constant finite row.

    Returns:
        np.float64 array of shape (num_instances, num_features): Standardized rows.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        # An over- or underflow here is detected below rather than reported to the caller.
        stds = X.std(axis=1, keepdims=True)  # (num_instances, 1)
    degenerate = (stds == 0.0) | np.isinf(stds)
    if degenerate.any():
        # Squaring the deviations underflows a tiny row's standard deviation to zero and overflows
        # a huge row's to infinity. Dividing such a row by its largest feature first keeps the
        # deviations in range, and standardizing is invariant to that scaling.
        rescale = degenerate & np.isfinite(X).all(axis=1, keepdims=True)
        if rescale.any():
            X = X / np.where(rescale, np.abs(X).max(axis=1, keepdims=True), 1.0)
            stds = X.std(axis=1, keepdims=True)

    return np.asarray((X - X.mean(axis=1, keepdims=True)) / stds, dtype=np.float64)


class InstanceStandardScaler(BaseTransformer):
    """Scale each feature vector to zero mean and unit standard deviation.

    Compute the mean and standard deviation across the features of each instance.
    Subtract the mean, then divide by the standard deviation.
    Constant finite vectors have zero variance and return zeros, including one-feature inputs.
    Vectors whose features are large or small enough to over- or underflow the standard deviation
    are rescaled before it is taken, so they standardize instead of returning zeros or infinities.
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

        return _standardize_rows(X[None, :])[0]

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
                return _standardize_rows(X)
            out = np.zeros(X.shape, dtype=np.float64)
            out[~const_finite] = _standardize_rows(X[~const_finite])
            return out
        return super().transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Scale each row of a 2-D feature batch to zero mean and unit standard deviation."""
        return self.transform(X)
