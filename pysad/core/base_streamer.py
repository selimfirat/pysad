import abc
from abc import abstractmethod
from collections.abc import Iterator
from typing import Any, Generic, TypeVar, overload

import numpy as np

_Features = TypeVar("_Features")
_Labels = TypeVar("_Labels")


class BaseStreamer(abc.ABC, Generic[_Features, _Labels]):
    """Abstract base class to simulate the streaming data.

    Args:
        shuffle (bool): Whether to shuffle the data initially (Default=False).
    """

    def __init__(self, shuffle: bool = False) -> None:
        self.shuffle = shuffle

    @overload
    def iter(self, X: _Features, y: None = None) -> Iterator[np.ndarray]: ...

    @overload
    def iter(self, X: _Features, y: _Labels) -> Iterator[tuple[np.ndarray, Any]]: ...

    @abstractmethod
    def iter(
        self, X: _Features, y: _Labels | None = None
    ) -> Iterator[np.ndarray | tuple[np.ndarray, Any]]:
        """Method that iterates array of data and (optionally) labels.

        Args:
            X: Feature rows in the container that the streamer supports.
            y: Labels in the same row order, or None (Default=None).
        """
        pass
