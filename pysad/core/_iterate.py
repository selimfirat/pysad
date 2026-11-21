from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy.typing as npt


def _iterate(
    X: npt.NDArray[Any], y: npt.NDArray[Any] | None = None
) -> Iterator[tuple[npt.NDArray[Any], Any | None]]:
    """Iterates array of features and possibly labels.

    Args:
        X (np.array of shape (num_instances, num_features)): The features array.
        y (np.array of shape (num_instances,)): The array containing labels (Default=None).
    """
    if y is None:
        for xi in X:
            yield xi, None
        return

    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")
    for xi, yi in zip(X, y, strict=True):
        yield xi, yi
