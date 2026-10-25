from __future__ import annotations

import numpy as np

from pysad.core.base_transformer import BaseTransformer


class InstanceStandardScaler(BaseTransformer):
    """Scales each feature vector to zero mean and unit standard deviation across its features.

    The scaler computes the mean and the standard deviation per instance, over the features of that instance. It subtracts the mean from each feature and divides the result by the standard deviation. It learns nothing from previous instances.
    """

    def __init__(self) -> None:
        super().__init__(-1)

    def fit_partial(self, X: np.ndarray) -> InstanceStandardScaler:
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

        X_mean = X.mean()
        X_std = X.std()

        return (X - X_mean) / X_std
