from abc import abstractmethod
import sklearn
from pysad.transform.projection.base_projector import BaseProjector


class BaseSKLearnProjector(BaseProjector):

    def __init__(self, num_components):
        """Abstract base projector class to wrap the random sklearn projectors.

        Args:
            num_components: The number of dimensions that the target will be projected into.
        """
        super().__init__(num_components)


    @property
    @abstractmethod
    def _projector(self):
        """ Helper property to wrap sklearn projectors.

        """
        pass

    def fit_partial(self, X):
        """Fits particular (next) timestep's features to train the projector.

        Args:
            X: np.float array of shape (n_components,).
                Input feature vector.
        Returns:
            self: object
                The fitted projector.
        """
        return self

    def transform_partial(self, X):
        """Projects particular (next) timestep's vector to (possibly) lower dimensional space.

        Args:
            X: np.float array of shape (num_features,)
                Input feature vector.

        Returns:
            projected_X: np.float array of shape (num_components,)
                Projected feature vector.
        """
        x = X.reshape(1, -1)

        return self._projector.fit_transform(x).reshape(-1)


class GaussianRandomProjector(BaseSKLearnProjector):
    """Reduces dimensionality through Gaussian random projection. The components of the random matrix are drawn from N(0, 1 / n_components). This text is taken from the `Sklearn documentation`_.

    Args:
        n_components : int or 'auto', optional (default = 'auto')
            Dimensionality of the target projection space.

            n_components can be automatically adjusted according to the
            number of samples in the dataset and the bound given by the
            Johnson-Lindenstrauss lemma. In that case the quality of the
            embedding is controlled by the ``eps`` parameter.

            It should be noted that Johnson-Lindenstrauss lemma can yield
            very conservative estimated of the required number of components
            as it makes no assumption on the structure of the dataset.

        eps : strictly positive float, optional (default=0.1)
            Parameter to control the quality of the embedding according to
            the Johnson-Lindenstrauss lemma when n_components is set to
            'auto'.

            Smaller values lead to better embedding and higher number of
            dimensions (n_components) in the target projection space.

    """

    def __init__(self, n_components='auto', *, eps=0.1):
        super().__init__(n_components)

        self.projector = sklearn.random_projection.GaussianRandomProjection(n_components=n_components, eps=eps)


class SparseRandomProjector(BaseSKLearnProjector):
    """The wrapper method for Sklearn's SparseRandomProjection. Reduces dimensionality through Gaussian random projection. The components of the random matrix are drawn from N(0, 1 / n_components). This text is taken from the `Sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.SparseRandomProjection.html#sklearn.random_projection.SparseRandomProjection>`_.

    Parameters
        n_components : int or 'auto', optional (default = 'auto')
            Dimensionality of the target projection space.

            n_components can be automatically adjusted according to the
            number of samples in the dataset and the bound given by the
            Johnson-Lindenstrauss lemma. In that case the quality of the
            embedding is controlled by the ``eps`` parameter.

            It should be noted that Johnson-Lindenstrauss lemma can yield
            very conservative estimated of the required number of components
            as it makes no assumption on the structure of the dataset.

        eps : strictly positive float, optional (default=0.1)
            Parameter to control the quality of the embedding according to
            the Johnson-Lindenstrauss lemma when n_components is set to
            'auto'.

            Smaller values lead to better embedding and higher number of
            dimensions (n_components) in the target projection space.

    """

    def __init__(self, n_components='auto', density="auto", eps=0.1, **kwargs):
        super().__init__(n_components, **kwargs)

        self.projector = sklearn.random_projection.SparseRandomProjection(n_components=n_components, density=density, eps=eps, dense_output=True)
