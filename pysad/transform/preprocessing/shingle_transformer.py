from __future__ import annotations

import numbers
from collections import deque
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from pysad.core.base_transformer import BaseTransformer


class ShingleTransformer(BaseTransformer):
    """Turns each point into a shingle, which holds the point and the points just before it.

    A shingle is the last ``shingle_size`` points of the stream in one vector, oldest first. A detector that gets single points sees only the values of each point. With shingles, the detector also sees the recent history. It can then find a change of a pattern over time, for example a value that is normal at noon but not at midnight. The paper of :class:`~pysad.models.RobustRandomCutForest` uses shingles in the same way :cite:`guha2016robust`.

    At the start of the stream, the transformer has fewer earlier points than a shingle needs. Copies of the oldest point fill the missing positions, so the shingle has no artificial jump. Before the first fit, copies of the new point fill them.

    ``fit_transform_partial`` builds the shingle of a point from the point and the points before it. Then it adds the point to the history. Thus a shingle never holds a later point.

    Args:
        shingle_size (int): The number of points in a shingle. It must be at least 1. For points with ``d`` features, a shingle has ``shingle_size * d`` features. A size of 1 gives each point unchanged.

    Raises:
        TypeError: If ``shingle_size`` is not an int.
        ValueError: If ``shingle_size`` is less than 1.
    """

    shingle_size: int

    def __init__(self, shingle_size: int) -> None:
        if isinstance(shingle_size, bool) or not isinstance(shingle_size, numbers.Integral):
            raise TypeError(f"shingle_size must be an int, got {shingle_size!r}.")
        if shingle_size < 1:
            raise ValueError(f"shingle_size must be at least 1, got {shingle_size}.")
        super().__init__(0)
        self.shingle_size = int(shingle_size)
        self.recent: deque[np.ndarray] = deque(maxlen=self.shingle_size - 1)

    @property
    def full(self) -> bool:
        """Whether the history holds ``shingle_size - 1`` points, so that a shingle needs no copies."""
        return len(self.recent) == self.shingle_size - 1

    def _point(self, X: ArrayLike) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64)
        if x.ndim > 1 or x.size == 0:
            raise ValueError(
                f"ShingleTransformer expected a 1-D point, got an input of shape {np.shape(X)}."
            )
        x = x.ravel()
        if not np.all(np.isfinite(x)):
            raise ValueError("ShingleTransformer got a point with non-finite values.")
        if self.recent and x.shape != self.recent[0].shape:
            raise ValueError(
                f"ShingleTransformer got a point with {x.shape[0]} features after points with "
                f"{self.recent[0].shape[0]}."
            )
        return x

    def fit_partial(self, X: np.ndarray) -> ShingleTransformer:
        """Adds the next point to the history.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            object: self.

        Raises:
            ValueError: If the point is not a 1-D point, is empty, or holds non-finite
                values; or if it has another number of features than the earlier points.
        """
        self.recent.append(self._point(X).copy())
        return self

    def transform_partial(self, X: np.ndarray) -> np.ndarray:
        """Returns the shingle that ends with the next point.

        The shingle holds the last ``shingle_size - 1`` fitted points, and then the point. This method does not change the history.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (shingle_size * num_features,): The shingle, oldest point first.

        Raises:
            ValueError: If the point is not a 1-D point, is empty, or holds non-finite
                values; or if it has another number of features than the earlier points.
        """
        x = self._point(X)
        earlier = list(self.recent)
        missing = self.shingle_size - 1 - len(earlier)
        return np.concatenate([*[earlier[0] if earlier else x] * missing, *earlier, x])

    def fit_transform_partial(self, X: np.ndarray) -> np.ndarray:
        """Returns the shingle that ends with the next point. Then adds the point to the history.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (shingle_size * num_features,): The shingle, oldest point first.

        Raises:
            ValueError: If the point is not a 1-D point, is empty, or holds non-finite
                values; or if it has another number of features than the earlier points.
        """
        shingle = self.transform_partial(X)
        self.fit_partial(X)
        return shingle

    def _transform_all(
        self, X: np.ndarray, transform_partial: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        # The width of a shingle depends on the number of features of the input.
        X = np.asarray(X)
        self.output_dims = self.shingle_size * (1 if X.ndim == 1 else X.shape[1])
        return super()._transform_all(X, transform_partial)
