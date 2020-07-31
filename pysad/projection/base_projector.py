from abc import ABC, abstractmethod
from pysad.utils import _iterate


class BaseProjector(ABC):

    def __init__(self, num_components):
        """Abstract base class for online projection methods.

        Args:
            num_components: The number of dimensions that the target will be projected into.
        """
        self.num_components = num_components

    @abstractmethod
    def fit_partial(self, X):
        """Fits particular (next) timestep's features to train the projector.

        Args:
            X: np.float array of shape (n_components,).
                Input feature vector.
        Returns:
            self: object
                The fitted projector.
        """
        pass

    @abstractmethod
    def transform_partial(self, X):
        """Projects particular (next) timestep's vector to (possibly) lower dimensional space.

        Args:
            X: np.float array of shape (num_features,)
                Input feature vector.

        Returns:
            projected_X: np.float array of shape (num_components,)
                Projected feature vector.
        """
        pass

    def fit(self, X):
        """Shortcut method that iteratively applies fit_partial to all instances in the input.

        Args:
            X: np.float array of shape (num_instances, num_features)

        Returns:
            self: object
                The fitted projector.
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
            projected_X: np.float array of shape (num_components,)
                Projected feature vectors.
        """
        self.fit_partial(X)

        return self.transform_partial(X)

    def transform(self, X):
        """Shortcut method that iteratively applies transform_partial to all instances in order.

        Args:
            X: np.float array of shape (num_instances,num_features)
                Input feature vectors.

        Returns:
            projected_X: np.float array of shape (num_instances, num_components)
                Projected feature vectors.
        """
        for xi, _ in _iterate(X):
            yield self.transform_partial(xi)

    def fit_transform(self, X):
        """Shortcut method that iteratively applies fit_transform_partial to all instances in order.

        Args:
            X: np.float array of shape (num_instances,num_features)
                Input feature vectors.

        Returns:
            projected_X: np.float array of shape (num_instances, num_components)
                Projected feature vectors.
        """
        for xi, _ in _iterate(X):
            yield self.fit_transform_partial(xi)
