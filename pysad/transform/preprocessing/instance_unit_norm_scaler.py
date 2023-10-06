from pysad.core.base_transformer import BaseTransformer
import numpy as np


class InstanceUnitNormScaler(BaseTransformer):
    """A scaler that makes the instance feature vector's norm equal to 1, i.e., the unit vector.

    Args:
        pow (float): The power, for which the norm is calculated. pow=2 is equivalent to the euclidean distance.
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
            scaled_X (np.float64 array of shape (features,)): Scaled feature vector.
        """
        X_norm = np.linalg.norm(X, ord=self.pow)

        return X / X_norm
