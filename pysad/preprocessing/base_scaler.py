from abc import abstractmethod

from pysad.utils import _iterate


class BaseScaler:
    """Abstract base class for scaling preprocessors.

    """

    def fit_transform_partial(self, X):
        """Shortcut method that applies fit_partial and transform_partial, respectively.

        Args:
            X: np.float array of shape (num_features,)
                Input feature vector.

        Returns:
            scaled_X: np.float array of shape (num_features,)
                Scaled feature vector.
        """
        self.fit_partial(X)

        return self.transform_partial(X)

    @abstractmethod
    def fit_partial(self, X):
        """Fits particular (next) timestep's features to train the scaler.

        Args:
            X: np.float array of shape (num_features,).
                Input feature vector.
        Returns:
            self: object
                The fitted scaler.
        """
        pass

    @abstractmethod
    def transform_partial(self, X):
        """Scales particular (next) timestep's vector.

        Args:
            X: np.float array of shape (num_features,)
                Input feature vector.

        Returns:
            scaled_X: np.float array of shape (features,)
                Scaled feature vector.
        """
        pass

    def fit(self, X):
        """Shortcut method that iteratively applies fit_partial to all instances in the input.

        Args:
            X: np.float array of shape (num_instances, num_features)

        Returns:
            self: object
                The fitted scaler.
        """
        for xi in self._iterate(X):
            self.fit_partial(xi)

        return self

    def fit_transform_partial(self, X):
        """Shortcut method that iteratively applies fit_partial and transform_partial, respectively.

        Args:
            X: np.float array of shape (num_features,)
                Input feature vectors.

        Returns:
            scaled_X: np.float array of shape (num_features,)
                Scaled feature vectors.
        """
        self.fit_partial(X)

        return self.transform_partial(X)

    def transform(self, X):
        """Shortcut method that iteratively applies transform_partial to all instances in order.

        Args:
            X: np.float array of shape (num_instances,num_features)
                Input feature vectors.

        Returns:
            scaled_X: np.float array of shape (num_instances, num_features)
                Scaled feature vectors.
        """
        for xi, _ in _iterate(X):
            yield self.transform_partial(xi)

    def fit_transform(self, X):
        """Shortcut method that iteratively applies fit_transform_partial to all instances in order.

        Args:
            X: np.float array of shape (num_instances,num_features)
                Input feature vectors.

        Returns:
            scaled_X: np.float array of shape (num_instances, num_features)
                Scaled feature vectors.
        """
        for xi, _ in _iterate(X):
            yield self.fit_transform_partial(xi)
