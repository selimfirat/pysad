import abc
from abc import abstractmethod
from collections.abc import Iterator
from typing import Any, overload

import numpy as np


class BaseStreamer(abc.ABC):
    """Abstract base class to simulate the streaming data.

    Args:
        shuffle (bool): Whether to shuffle the data initially (Default=False).
    """

    def __init__(self, shuffle: bool = False) -> None:
        self.shuffle = shuffle

    @overload
    def iter(self, X: np.ndarray, y: None = None) -> Iterator[np.ndarray]: ...

    @overload
    def iter(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[np.ndarray, Any]]: ...

    @abstractmethod
    def iter(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> Iterator[np.ndarray | tuple[np.ndarray, Any]]:
        """Method that iterates array of data and (optionally) labels.

        Args:
            X (np.array of shape (num_instances, num_features)): The features of instances to iterate.
            y (np.array of shape (num_instances,) or None): If not None, the labels to iterate in the same order (Default=None).
        """
        pass
