from pysad.core.base_transformer import BaseTransformer


class InstanceStandardScaler(BaseTransformer):
    """Standard deviation scaling per instance. Not that the variance and mean is calculated per instance, for which the scaling is done with.
    The method substracts mean and divides with the standard deviation of the features, separately for each instance.
    """

    def __init__(self):
        super().__init__(-1)

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

        X_mean = X.mean()
        X_std = X.std()

        return (X - X_mean) / X_std
