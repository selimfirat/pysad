import abc
from abc import abstractmethod


class BaseStreamer(abc.ABC):
    """Abstract base class to simulate the streaming data.

    Args:
        shuffle (bool): Whether shuffle the data initially (Optional, default=False).
    """

    def __init__(self, shuffle=False):
        self.shuffle = shuffle

    @abstractmethod
    def iter(self, X, y=None):
        """Method that iterates array of data and (optionally) labels.

        Args:
            X (np.array of shape (num_instances, num_features)): The features of instances to iterate.
            y: (Optional, default=None) If not None, iterates labels with the same order.
        """
        pass
