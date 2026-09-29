from collections.abc import Iterator
from typing import Any, overload

import numpy as np

from pysad.core.base_streamer import BaseStreamer


class ArrayStreamer(BaseStreamer):
    """Simulator class to iterate array(s).

    Args:
        shuffle (bool): Whether shuffle the data initially (Default=False).
    """

    def __init__(self, shuffle: bool = False) -> None:
        self.shuffle = shuffle

    @overload
    def iter(self, X: np.ndarray, y: None = None) -> Iterator[np.ndarray]: ...

    @overload
    def iter(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[np.ndarray, Any]]: ...

    def iter(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> Iterator[np.ndarray | tuple[np.ndarray, Any]]:
        """Iterates array of features and possibly labels.

        Args:
            X (np.array of shape (num_instances, num_features)): The features array.
            y (np.array of shape (num_instances, ): The array containing labels (Default=None).
        """
        indices = list(range(len(X)))
        if self.shuffle:
            np.random.shuffle(indices)

        if y is None:
            for i in indices:
                yield X[i]
        else:
            if len(X) != len(y):
                raise ValueError("X and y must have the same length.")
            for i in indices:
                yield X[i], y[i]
