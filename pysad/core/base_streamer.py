import abc
from abc import abstractmethod
from collections.abc import Iterator

import numpy as np


class BaseStreamer(abc.ABC):
    """Abstract base class to simulate the streaming data.

    Args:
        shuffle (bool): Whether shuffle the data initially (Optional, default=False).
    """

    def __init__(self, shuffle: bool = False) -> None:
        self.shuffle = shuffle

    @abstractmethod
    def iter(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> Iterator[np.ndarray | tuple[np.ndarray, np.ndarray]]:
        """Method that iterates array of data and (optionally) labels.

        Args:
            X (np.array of shape (num_instances, num_features)): The features of instances to iterate.
            y: (Optional, default=None) If not None, iterates labels with the same order.
        """
        pass