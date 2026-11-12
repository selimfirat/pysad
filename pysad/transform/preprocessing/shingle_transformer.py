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

    def _validate_batch(self, X: np.ndarray) -> np.ndarray | None:
        if not isinstance(X, np.ndarray) or X.dtype == object or X.ndim == 0:
            return None
        if X.ndim > 2:
            raise ValueError(
                f"ShingleTransformer expected a 1-D point, got an input of shape {X.shape[1:]}."
            )
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n, d = arr.shape
        if n > 0 and d == 0:
            raise ValueError("ShingleTransformer expected a 1-D point, got an input of shape (0,).")
        if n > 0:
            if not np.isfinite(arr).all():
                raise ValueError("ShingleTransformer got a point with non-finite values.")
            if self.recent and d != self.recent[0].shape[0]:
                raise ValueError(
                    f"ShingleTransformer got a point with {d} features after points with "
                    f"{self.recent[0].shape[0]}."
                )
        return arr

    def fit(self, X: np.ndarray) -> ShingleTransformer:
        """Adds all points in the batch to the history, keeping the last ``shingle_size - 1``."""
        arr = self._validate_batch(X)
        if arr is None:
            return super().fit(X)  # type: ignore[return-value]
        k = self.shingle_size
        if k > 1 and len(arr) > 0:
            for row in arr[-(k - 1) :]:
                self.recent.append(row.copy())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Returns the shingles for all points in the batch without changing the history."""
        arr = self._validate_batch(X)
        if arr is None:
            return super().transform(X)
        n, d = arr.shape
        k = self.shingle_size
        self.output_dims = k * d
        if n == 0:
            return np.empty((0, k * d), dtype=np.float64)
        if k == 1:
            return arr.copy()
        earlier = list(self.recent)
        m = len(earlier)
        if m == 0:
            return np.tile(arr, (1, k))
        missing = k - 1 - m
        prefix = np.concatenate([*[earlier[0]] * missing, *earlier])
        out = np.empty((n, k * d), dtype=np.float64)
        out[:, : (k - 1) * d] = prefix
        out[:, (k - 1) * d :] = arr
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Builds shingles for all points in stream order and updates the history."""
        arr = self._validate_batch(X)
        if arr is None:
            return super().fit_transform(X)
        n, d = arr.shape
        k = self.shingle_size
        self.output_dims = k * d
        if n == 0:
            return np.empty((0, k * d), dtype=np.float64)
        if k == 1:
            return arr.copy()
        earlier = list(self.recent)
        m = len(earlier)
        missing = k - 1 - m
        padded = np.empty((n + k - 1, d), dtype=np.float64)
        if missing > 0:
            padded[:missing] = earlier[0] if m > 0 else arr[0]
        if m > 0:
            padded[missing : k - 1] = earlier
        padded[k - 1 :] = arr
        windows = np.lib.stride_tricks.sliding_window_view(padded, k, axis=0)  # (n, d, k)
        out = np.swapaxes(windows, 1, 2).reshape(n, k * d)
        if not out.flags.owndata:
            out = out.copy()
        for row in arr[-(k - 1) :]:
            self.recent.append(row.copy())
        return out

    def _transform_all(
        self, X: np.ndarray, transform_partial: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        # The width of a shingle depends on the number of features of the input.
        X = np.asarray(X)
        self.output_dims = self.shingle_size * (1 if X.ndim == 1 else X.shape[1])
        return super()._transform_all(X, transform_partial)
