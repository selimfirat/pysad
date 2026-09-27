import abc
from abc import abstractmethod
from collections.abc import Iterator
from typing import Any, overload

import numpy as np


class BaseStreamer(abc.ABC):
    """Abstract base class to simulate the streaming data.

    Args:
        shuffle (bool): Whether shuffle the data initially (Optional, default=False).
    """

    def __init__(self, shuffle: bool = False) -> None:
        self.shuffle = shuffle

    @overload
    def iter(
        self, X: np.ndarray, y: None = None
    ) -> Iterator[np.ndarray]:
        ...

    @overload
    def iter(
        self, X: np.ndarray, y: np.ndarray
    ) -> Iterator[tuple[np.ndarray, Any]]:
        ...

    @abstractmethod
    def iter(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> Iterator[np.ndarray | tuple[np.ndarray, Any]]:
        """Method that iterates array of data and (optionally) labels.

        Args:
            X (np.array of shape (num_instances, num_features)): The features of instances to iterate.
            y: (Optional, default=None) If not None, iterates labels with the same order.
        """
        pass
