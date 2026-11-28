from __future__ import annotations

from collections.abc import Iterable, Iterator, Sized
from typing import Any


def _iterate(X: Iterable[Any], y: Iterable[Any] | None = None) -> Iterator[tuple[Any, Any | None]]:
    """Iterates sequence or iterable of features and possibly labels.

    Args:
        X (np.array of shape (num_instances, num_features) or iterable of instances): The features input.
        y (np.array of shape (num_instances,) or sequence of labels): The labels input (Default=None).
    """
    if y is None:
        for xi in X:
            yield xi, None
        return

    x_seq = X if isinstance(X, Sized) else list(X)
    y_seq = y if isinstance(y, Sized) else list(y)
    if len(x_seq) != len(y_seq):
        raise ValueError("X and y must have the same length.")
    for xi, yi in zip(x_seq, y_seq, strict=True):
        yield xi, yi
