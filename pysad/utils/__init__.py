"""
The :mod:`pysad.utils` module includes utility functions used in the `PySAD` framework, which can also be useful for streaming learning.
"""

import random

import numpy as np

from .array_streamer import ArrayStreamer
from .array_streamer import _iterate as _iterate
from .csv_streamer import CSVStreamer
from .data import Data
from .pandas_streamer import PandasStreamer
from .plotting import plot_stream
from .polars_streamer import PolarsStreamer
from .synthetic import generate_stream
from .validation import check_feature_bounds, check_univariate
from .window import RingBuffer, Window

__all__ = [
    "ArrayStreamer",
    "check_feature_bounds",
    "check_univariate",
    "CSVStreamer",
    "Data",
    "fix_seed",
    "generate_stream",
    "get_minmax_array",
    "get_minmax_scalar",
    "PandasStreamer",
    "plot_stream",
    "PolarsStreamer",
    "RingBuffer",
    "Window",
]


def fix_seed(seed: int) -> None:
    """Seed Python's ``random`` module and NumPy's global random state.

    Components with ``random_state=None`` draw from NumPy's global state.

    Args:
        seed (int): Seed used to reproduce random draws.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeding the global state is the point of this function.


def get_minmax_array(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the minimum and maximum of each feature across input instances.

    The returned arrays can be passed as ``feature_mins`` and ``feature_maxes``
    to ``HalfSpaceTrees`` and ``RSHash``.

    Args:
        X (np.float64 array of shape (num_instances, num_features)): The input array.

    Returns:
        tuple[np.ndarray, np.ndarray]: The minimum and the maximum of each feature, each a np.float64 array of shape (num_features,).
    """
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)

    return min, max


def get_minmax_scalar(x: np.ndarray) -> tuple[float, float]:
    """Return the minimum and maximum across all elements as Python floats.

    Args:
        x (np.float64 array of any shape): Input array.

    Returns:
        tuple[float, float]: The minimum and the maximum value in the array.
    """
    return float(x.min()), float(x.max())
