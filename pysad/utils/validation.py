from __future__ import annotations

import math
import numbers
from typing import Any

import numpy as np


def _int_at_least(value: Any, name: str, minimum: int, expected: str = "an int") -> int:
    """Returns value as a Python int.

    If value is a bool or not an integer, this function raises TypeError.
    If value is below minimum, this function raises ValueError.

    Args:
        value (object): The value to check.
        name (str): The parameter name for the error message.
        minimum (int): The smallest accepted value.
        expected (str): What the parameter accepts, for the error message (Default="an int").

    Returns:
        int: The value as a Python int.

    Raises:
        TypeError: If value is a bool or not an integer.
        ValueError: If value is below minimum.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be {expected}, got {value!r}.")

    value_int = int(value)
    if value_int < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value_int}.")

    return value_int


def _finite_float(value: Any, name: str) -> float:
    """Returns value as a Python float.

    If value is a bool or not a real number, this function raises TypeError.
    If value is not finite, this function raises ValueError.

    Args:
        value (object): The value to check.
        name (str): The parameter name for the error message.

    Returns:
        float: The value as a Python float.

    Raises:
        TypeError: If value is a bool or not a real number.
        ValueError: If value is NaN or infinite.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}.")

    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"{name} must be finite, got {value_float}.")

    return value_float


def check_feature_bounds(feature_mins: Any, feature_maxes: Any) -> tuple[np.ndarray, np.ndarray]:
    """Converts and validates feature minimum and maximum boundary arrays.

    The boundary arrays must be non-empty 1-D arrays of the same length.
    All values must be finite.
    Every minimum must not exceed its maximum.

    Args:
        feature_mins (array-like): Minimum boundary of the features.
        feature_maxes (array-like): Maximum boundary of the features.

    Returns:
        tuple[np.ndarray, np.ndarray]: The bounds as float64 1-D arrays.

    Raises:
        ValueError: If the arrays are empty, have different shapes, are not 1-D, have non-finite values, or have a minimum above its maximum.
    """
    mins = np.array(feature_mins, dtype=np.float64)
    maxes = np.array(feature_maxes, dtype=np.float64)
    if mins.ndim != 1 or mins.shape != maxes.shape or mins.size == 0:
        raise ValueError(
            "feature_mins and feature_maxes must be non-empty 1-D arrays of the same length."
        )
    if not (np.all(np.isfinite(mins)) and np.all(np.isfinite(maxes))):
        raise ValueError("feature_mins and feature_maxes must be finite.")
    if np.any(mins > maxes):
        raise ValueError("feature_mins must not exceed feature_maxes.")

    return mins, maxes


def check_univariate(X: Any, name: str = "Model", *, reject_nan: bool = False) -> float:
    """Returns the single value of a univariate instance.

    If the instance does not hold exactly one value, this function raises ValueError.
    If reject_nan is True and the value is NaN, this function raises ValueError.

    Args:
        X (float or array-like): The instance, holding exactly one value.
        name (str): The model name for the error message (Default="Model").
        reject_nan (bool): Whether to reject NaN values (Default=False).

    Returns:
        float: The value of the instance as a float.

    Raises:
        ValueError: If the instance does not hold exactly one value, or if reject_nan is True and the value is NaN.
    """
    values = np.asarray(X, dtype=np.float64).ravel()
    if values.size != 1:
        raise ValueError(
            f"{name} is univariate: expected an instance with one value, "
            f"got an instance of shape {np.shape(X)}."
        )

    val = float(values[0])
    if reject_nan and math.isnan(val):
        raise ValueError(f"{name} does not accept NaN values.")

    return val
