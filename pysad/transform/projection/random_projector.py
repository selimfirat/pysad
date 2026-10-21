from abc import abstractmethod
from typing import Any

import numpy as np
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from pysad.core.base_transformer import BaseTransformer


class BaseSKLearnProjector(BaseTransformer):
    def __init__(self, num_components: int | str) -> None:
        """Abstract base projector class to wrap the random sklearn projectors.

        Args:
            num_components (int or 'auto'): The number of dimensions that the target will be projected into.
        """
        super().__init__(num_components)  # type: ignore[arg-type]
        self._components: Any = None

    @property
    @abstractmethod
    def _projector(self) -> GaussianRandomProjection | SparseRandomProjection:
        """Helper property to wrap sklearn projectors."""
        pass

    def _fit_projector(self, X: np.ndarray) -> None:
        """Draws the random projection matrix once, sized from the given instances.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): Input feature vectors.
        """
        if getattr(self, "num_components", None) == "auto" and X.shape[0] == 1:
            raise ValueError(
                "num_components='auto' is sized from the number of instances, so it cannot be resolved "
                "from a single instance. Call fit(X) on a batch first or set num_components to an int."
            )

        self._components = self._projector().fit(X).components_
        self.output_dims = self._components.shape[0]

    def fit(self, X: np.ndarray) -> "BaseSKLearnProjector":
        """Draws the random projection matrix from all instances, which also resolves num_components='auto'.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): Input feature vectors.

        Returns:
            object: self.
        """
        if self._components is None:
            self._fit_projector(X)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projects all instances, fitting the projector to them first if it is not fitted yet.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): Input feature vectors.

        Returns:
            np.float64 array of shape (num_instances, num_components): Projected feature vectors.
        """
        self.fit(X)

        return super().transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits the projector to all instances, then projects them.

        Args:
            X (np.float64 array of shape (num_instances, num_features)): Input feature vectors.

        Returns:
            np.float64 array of shape (num_instances, num_components): Projected feature vectors.
        """
        return self.fit(X).transform(X)

    def fit_partial(self, X: np.ndarray) -> "BaseSKLearnProjector":
        """Fits particular (next) timestep's features to train the projector.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            object: self.
        """
        if self._components is None:
            self._fit_projector(X.reshape(1, -1))

        return self

    def transform_partial(self, X: np.ndarray) -> np.ndarray:
        """Projects particular (next) timestep's vector to (possibly) lower dimensional space.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (num_components,): Projected feature vector.
        """
        if self._components is None:
            self._fit_projector(X.reshape(1, -1))

        assert self._components is not None
        return self._components.dot(X)


class GaussianRandomProjector(BaseSKLearnProjector):
    """Reduces dimensionality through Gaussian random projection."""

    def __init__(
        self,
        num_components: int | str = "auto",
        *,
        eps: float = 0.1,
        random_state: Any = None,
    ) -> None:
        super().__init__(num_components)
        self.eps = eps
        self.num_components = num_components
        self.random_state = random_state

    def _projector(self) -> GaussianRandomProjection:
        return GaussianRandomProjection(
            n_components=self.num_components, eps=self.eps, random_state=self.random_state
        )


class SparseRandomProjector(BaseSKLearnProjector):
    """Reduces dimensionality through sparse random projection."""

    def __init__(
        self,
        num_components: int | str = "auto",
        density: float | str = "auto",
        eps: float = 0.1,
        random_state: Any = None,
    ) -> None:
        super().__init__(num_components)
        self.eps = eps
        self.density = density
        self.num_components = num_components
        self.random_state = random_state

    def _projector(self) -> SparseRandomProjection:
        return SparseRandomProjection(
            n_components=self.num_components,
            density=self.density,
            eps=self.eps,
            dense_output=True,
            random_state=self.random_state,
        )
