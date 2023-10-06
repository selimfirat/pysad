from pysad.core.base_transformer import BaseTransformer


class IdentityScaler(BaseTransformer):
    """A scaler that does not modify the input, which is added for convenience.
    """

    def __init__(self):
        super().__init__(-1)

    def fit_partial(self, X):
        """Convenience method that does not modify the input or the scaler.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            object: The scaler.
        """
        return self

    def transform_partial(self, X):
        """Convenience method that does not modify the input.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            X (np.float64 array of shape (features,)): The exact same input feature vector.
        """
        return X
