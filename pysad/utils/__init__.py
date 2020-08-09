from pysad.utils.array_streamer import ArrayStreamer
import random
import numpy as np


def fix_seed(seed):
    """Utility method to fix the seed for randomness.

    Args:
        seed: The seed.
    """
    random.seed(seed)
    np.random.seed(seed)


def get_minmax_array(X):
    """Utility method that returns the boundaries for each feature of the input array.

    Args:
        X: np.float array of shape (num_instances, num_features)
            The input array.
    Returns:
        min: np.float array of shape (num_features,)
            Minimum values for each feature in array.
        max: np.float array of shape (num_features,)
            Maximum values for each feature in array.
    """
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)

    return min, max


def get_minmax_scalar(x):
    """Utility method that returns the boundaries of the input array.

    Args:
        X: np.float array of any shape.
    Returns:
        min: float
            Minimum value in array.
        max: float
            Maximum value in array.
    """
    min = np.min(x)
    max = np.max(x)

    return min, max


def _iterate(X, y=None):
    iterator = ArrayStreamer(shuffle=False)

    if y is None:
        for xi in iterator.iter(X):
            yield xi, None
    else:
        for xi, yi in iterator.iter(X, y):
            yield xi, yi
