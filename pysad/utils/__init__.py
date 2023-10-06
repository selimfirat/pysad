"""
The :mod:`pysad.utils` module includes utility functions used in the `PySAD` framework, which can also be useful streaming learning.
"""
from .array_streamer import ArrayStreamer
import random
import numpy as np
from .data import Data
from .pandas_streamer import PandasStreamer
from .window import Window

__all__ = ["fix_seed", "get_minmax_array", "get_minmax_scalar", "_iterate", "ArrayStreamer", "PandasStreamer", "Window", "Data"]


def fix_seed(seed):
    """Utility method to fix the seed for randomness.

    Args:
        seed (int): The seed.
    """
    random.seed(seed)
    np.random.seed(seed)


def get_minmax_array(X):
    """Utility method that returns the boundaries for each feature of the input array.

    Args:
        X (np.float64 array of shape (num_instances, num_features)): The input array.
    Returns:
        min (np.float64 array of shape (num_features,)): Minimum values for each feature in array.
        max (np.float64 array of shape (num_features,)): Maximum values for each feature in array.
    """
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)

    return min, max


def get_minmax_scalar(x):
    """Utility method that returns the boundaries of the input array.

    Args:
        X (np.float64 array of any shape): Input array.
    Returns:
        float: Minimum value in array.
        float: Maximum value in array.
    """
    min = np.min(x)
    max = np.max(x)

    return min, max


def _iterate(X, y=None):
    """Iterates array of features and possibly labels.

    Args:
        X (np.array of shape (num_instances, num_features)): The features array.
        y (np.array of shape (num_instances, ): The array containing labels (Default=None).
    """

    iterator = ArrayStreamer(shuffle=False)

    if y is None:
        for xi in iterator.iter(X):
            yield xi, None
    else:
        for xi, yi in iterator.iter(X, y):
            yield xi, yi
