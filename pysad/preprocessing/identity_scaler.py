from pysad.preprocessing.base_scaler import BaseScaler


class IdentityScaler(BaseScaler):
    """A scaler that does not modify the input, which is added for convenience.
    """

    def fit_partial(self, X):
        """Convenience method that does not modify the input or the scaler.

        Args:
            X: np.float array of shape (num_features,).
                Input feature vector.
        Returns:
            self: object
                The scaler.
        """
        return self

    def transform_partial(self, X):
        """Convenience method that does not modify the input.

        Args:
            X: np.float array of shape (num_features,)
                Input feature vector.

        Returns:
            X: np.float array of shape (features,)
                The exact same input feature vector.
        """
        return X
