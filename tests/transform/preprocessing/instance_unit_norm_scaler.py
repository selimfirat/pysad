from pysad.core.base_transformer import BaseTransformer


class InstanceUnitNormScaler(BaseTransformer):
    """A scaler that makes the instance feature vector's norm equal to 1, i.e., the unit vector.

    Args:
        pow: The power, for which the norm is calculated. pow=2 is equivalent to the euclidean distance.
    """

    def __init__(self, pow=2):
        self.pow = pow

    def fit_partial(self, X):
        """Fits particular (next) timestep's features to train the scaler.

        Args:
            X: np.float array of shape (num_features,).
                Input feature vector.
        Returns:
            self: object
                The fitted scaler.
        """
        return self

    def transform_partial(self, X):
        """Scales particular (next) timestep's vector.

        Args:
            X: np.float array of shape (num_features,)
                Input feature vector.

        Returns:
            scaled_X: np.float array of shape (features,)
                Scaled feature vector.
        """
        self.output_dims = X.shape[0]
        X_norm = X.norm(p=self.pow, dim=1, keepdim=True)

        return X / X_norm.expand_as(X)
