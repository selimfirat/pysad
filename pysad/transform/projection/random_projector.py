from abc import abstractmethod
import numpy as np
from scipy.sparse import issparse
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from pysad.core.base_transformer import BaseTransformer


class BaseSKLearnProjector(BaseTransformer):

    def __init__(self, num_components):
        """Abstract base projector class to wrap the random sklearn projectors.

        Args:
            num_components (int): The number of dimensions that the target will be projected into.
        """
        super().__init__(num_components)
        self._components = None

    @property
    @abstractmethod
    def _projector(self):
        """ Helper property to wrap sklearn projectors.

        """
        pass

    def _fit_projector(self, x):
        """Draws the random projection matrix once, from the first instance's dimensionality.

        Args:
            x (np.float64 array of shape (1, num_features)): Input feature vector.
        """
        components = self._projector().fit(x).components_
        self._components = components.toarray() if issparse(components) else components

    def fit_partial(self, X):
        """Fits particular (next) timestep's features to train the projector.

        Args:
            X (np.float64 array of shape (num_components,)): Input feature vector.
        Returns:
            object: self.
        """
        if self._components is None:
            self._fit_projector(X.reshape(1, -1))

        return self

    def transform_partial(self, X):
        """Projects particular (next) timestep's vector to (possibly) lower dimensional space.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            projected_X: np.float64 array of shape (num_components,)
                Projected feature vector.
        """
        x = X.reshape(1, -1)
        if self._components is None:
            self._fit_projector(x)

        # np.dot rather than sklearn's transform: its matmul raises spurious FPE warnings with macOS Accelerate BLAS.
        return np.dot(x, self._components.T).reshape(-1)


class GaussianRandomProjector(BaseSKLearnProjector):
    """Reduces dimensionality through Gaussian random projection. The components of the random matrix are drawn from N(0, 1 / n_components). This text is taken from the `Sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html#sklearn.random_projection.GaussianRandomProjection>`_.

    Args:
        n_components (int or 'auto'): Dimensionality of the target projection space, optional (default = 'auto').

            n_components can be automatically adjusted according to the
            number of samples in the dataset and the bound given by the
            Johnson-Lindenstrauss lemma. In that case the quality of the
            embedding is controlled by the ``eps`` parameter.

            It should be noted that Johnson-Lindenstrauss lemma can yield
            very conservative estimated of the required number of components
            as it makes no assumption on the structure of the dataset.

        eps (strictly positive float, optional): (default=0.1)
            Parameter to control the quality of the embedding according to
            the Johnson-Lindenstrauss lemma when n_components is set to
            'auto'.

            Smaller values lead to better embedding and higher number of
            dimensions (n_components) in the target projection space.

    """

    def __init__(self, num_components='auto', *, eps=0.1):
        super().__init__(num_components)
        self.eps = eps
        self.num_components = num_components

    def _projector(self):
        return GaussianRandomProjection(
            n_components=self.num_components, eps=self.eps)


class SparseRandomProjector(BaseSKLearnProjector):
    """The wrapper method for Sklearn's SparseRandomProjection. Reduces dimensionality through Gaussian random projection. The components of the random matrix are drawn from N(0, 1 / n_components). This text is taken from the `Sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.SparseRandomProjection.html#sklearn.random_projection.SparseRandomProjection>`_.

    Parameters
        n_components (int or 'auto'): Optional (default = 'auto')
            Dimensionality of the target projection space.

            n_components can be automatically adjusted according to the
            number of samples in the dataset and the bound given by the
            Johnson-Lindenstrauss lemma. In that case the quality of the
            embedding is controlled by the ``eps`` parameter.

            It should be noted that Johnson-Lindenstrauss lemma can yield
            very conservative estimated of the required number of components
            as it makes no assumption on the structure of the dataset.

        eps (strictly positive float): Optional (default=0.1)
            Parameter to control the quality of the embedding according to
            the Johnson-Lindenstrauss lemma when n_components is set to
            'auto'.

            Smaller values lead to better embedding and higher number of
            dimensions (n_components) in the target projection space.

    """

    def __init__(self, num_components='auto', density="auto", eps=0.1):
        super().__init__(num_components)
        self.eps = eps
        self.density = density
        self.num_components = num_components

    def _projector(self):
        return SparseRandomProjection(
            n_components=self.num_components,
            density=self.density,
            eps=self.eps,
            dense_output=True)
