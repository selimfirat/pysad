from pysad.core.base_streamer import BaseStreamer
import numpy as np


class ArrayStreamer(BaseStreamer):
    """Simulator class to iterate array(s).

    Args:
        shuffle: Whether shuffle the data initially.
    """

    def __init__(self, shuffle=False):
        self.shuffle = shuffle

    def iter(self, X, y=None):
        """Iterates array of features and possibly labels.

        Args:
            X: The features array, which can also be a numpy array
            y: The array containing labels.
        """
        indices = list(range(len(X)))
        if self.shuffle:
            np.random.shuffle(indices)

        if y is None:
            for i in indices:
                yield X[i]
        else:
            assert len(X) == len(y)
            for i in indices:
                yield X[i], y[i]
