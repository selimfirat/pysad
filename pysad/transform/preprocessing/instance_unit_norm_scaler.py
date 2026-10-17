import numpy as np

from pysad.core.base_transformer import BaseTransformer


class InstanceUnitNormScaler(BaseTransformer):
    """Scales each feature vector to unit norm.

    Args:
        pow (float): The order of the norm. ``pow=2`` gives the Euclidean norm (Default=2).
    """

    def __init__(self, pow=2):
        super().__init__(-1)
        self.pow = pow

    def fit_partial(self, X):
        """Fits particular (next) timestep's features to train the scaler.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            object: self.
        """
        return self

    def transform_partial(self, X):
        """Scales particular (next) timestep's vector.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (num_features,): Scaled feature vector.
        """
        X_norm = np.linalg.norm(X, ord=self.pow)

        return X / X_norm
