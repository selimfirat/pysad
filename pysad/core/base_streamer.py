import abc
from abc import abstractmethod


class BaseStreamer(abc.ABC):
    """Abstract base class to simulate the streaming data

    Args:
        shuffle: Whether shuffle the data initially.
    """

    def __init__(self, shuffle=False):
        self.shuffle = shuffle

    @abstractmethod
    def iter(self, X, y=None):
        """Method that iterates array of data and labels

        Args:
            X: The features.
            y: (Optional, default=None) If not None, iterates labels with the same order.
        """
        pass
