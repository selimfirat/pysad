from collections.abc import Iterator
from typing import Any, overload

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_streamer import BaseStreamer


class ArrayStreamer(BaseStreamer):
    """Provide a stream iterator over feature arrays and optional labels.

    Args:
        shuffle (bool): Whether to shuffle the data initially (Default=False).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the shuffling. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self, shuffle: bool = False, random_state: int | np.random.RandomState | None = None
    ) -> None:
        self.shuffle = shuffle
        self.random_state = random_state

    @overload
    def iter(self, X: np.ndarray, y: None = None) -> Iterator[np.ndarray]: ...

    @overload
    def iter(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[np.ndarray, Any]]: ...

    def iter(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> Iterator[np.ndarray | tuple[np.ndarray, Any]]:
        """Iterate over feature rows, optionally paired with labels.

        Args:
            X (np.ndarray): Feature array of shape
                (num_instances, num_features).
            y (np.ndarray or None): Optional labels with shape
                (num_instances, ...). Defaults to None.

        Yields:
            np.ndarray: If y is None, a feature row of shape
                (num_features,).
            tuple[np.ndarray, Any]: If y is provided, a pair
                (X[i], y[i]), where X[i] has shape (num_features,)
                and y[i] retains its original per-instance shape.
        """
        indices: range | list[int] = range(len(X))
        if self.shuffle:
            indices = list(indices)
            check_random_state(self.random_state).shuffle(indices)

        if y is None:
            for i in indices:
                yield X[i]
        else:
            if len(X) != len(y):
                raise ValueError("X and y must have the same length.")
            for i in indices:
                yield X[i], y[i]


def _iterate(X: np.ndarray, y: np.ndarray | None = None) -> Iterator[tuple[np.ndarray, Any | None]]:
    """Iterates array of features and possibly labels.

    Args:
        X (np.array of shape (num_instances, num_features)): The features array.
        y (np.array of shape (num_instances,)): The array containing labels (Default=None).
    """
    iterator = ArrayStreamer(shuffle=False)

    if y is None:
        for xi in iterator.iter(X):
            yield xi, None
    else:
        for xi, yi in iterator.iter(X, y):
            yield xi, yi
