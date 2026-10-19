import numbers
from collections import deque

import numpy as np

from pysad.core.base_transformer import BaseTransformer


class ShingleTransformer(BaseTransformer):
    """Turns each point into a shingle: the last ``shingle_size`` points of the stream side by side, oldest first.

    A detector of single points sees only the values of each point. With shingles, it also sees the recent history, so it can find a change of a pattern over time, for example a value that is normal at noon but not at midnight. The paper of :class:`~pysad.models.RobustRandomCutForest` shingles its streams in this way :cite:`guha2016robust`.

    Until ``shingle_size - 1`` points are fitted, a shingle has too few earlier points. Then the oldest point that the transformer has, or the new point itself, fills the missing positions, so that a shingle has no artificial jump. ``fit_transform_partial`` builds the shingle of a point from the points before it, and then fits the point.

    Args:
        shingle_size (int): The number of consecutive points in a shingle, at least 1. A shingle of a point with ``d`` features has ``shingle_size * d`` features. 1 gives each point as it is.

    Raises:
        TypeError: If ``shingle_size`` is not an int.
        ValueError: If ``shingle_size`` is less than 1.
    """

    def __init__(self, shingle_size):
        if isinstance(shingle_size, bool) or not isinstance(shingle_size, numbers.Integral):
            raise TypeError(f"shingle_size must be an int, got {shingle_size!r}.")
        if shingle_size < 1:
            raise ValueError(f"shingle_size must be at least 1, got {shingle_size}.")
        super().__init__(0)
        self.shingle_size = int(shingle_size)
        self.recent = deque(maxlen=self.shingle_size - 1)

    def _point(self, X):
        x = np.ravel(np.asarray(X, dtype=np.float64))
        if self.recent and x.shape != self.recent[0].shape:
            raise ValueError(
                f"ShingleTransformer got a point with {x.shape[0]} features after points with "
                f"{self.recent[0].shape[0]}."
            )
        return x

    def fit_partial(self, X):
        """Adds the next point to the history of the shingles.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            object: self.

        Raises:
            ValueError: If the point has a different number of features than the earlier points.
        """
        self.recent.append(self._point(X).copy())
        return self

    def transform_partial(self, X):
        """Returns the shingle that ends with the next point: the last ``shingle_size - 1`` fitted points, then the point.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (shingle_size * num_features,): The shingle, oldest point first.

        Raises:
            ValueError: If the point has a different number of features than the earlier points.
        """
        x = self._point(X)
        earlier = list(self.recent)
        missing = self.shingle_size - 1 - len(earlier)
        return np.concatenate([*[earlier[0] if earlier else x] * missing, *earlier, x])

    def fit_transform_partial(self, X):
        """Returns the shingle that ends with the next point, and then adds the point to the history.

        Args:
            X (np.float64 array of shape (num_features,)): Input feature vector.

        Returns:
            np.float64 array of shape (shingle_size * num_features,): The shingle, oldest point first.

        Raises:
            ValueError: If the point has a different number of features than the earlier points.
        """
        shingle = self.transform_partial(X)
        self.fit_partial(X)
        return shingle

    def _transform_all(self, X, transform_partial):
        # The width of a shingle depends on the number of features of the input.
        X = np.asarray(X)
        self.output_dims = self.shingle_size * (1 if X.ndim == 1 else X.shape[1])
        return super()._transform_all(X, transform_partial)
