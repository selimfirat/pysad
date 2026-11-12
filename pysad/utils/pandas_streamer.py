from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from pysad.core.base_streamer import BaseStreamer
from pysad.utils.array_streamer import ArrayStreamer

if TYPE_CHECKING:
    import pandas as pd


def _iter_rows(data: pd.DataFrame | pd.Series) -> Iterator[Any]:
    dtype = data.iloc[:0].to_numpy().dtype
    if dtype.kind in "biu":
        column = data if data.ndim == 1 else data.iloc[:, 0]
        # NumPy integer and boolean columns cannot contain missing values.
        # Scan nullable columns in chunks to bound temporary memory.
        if not isinstance(column.dtype, np.dtype) and any(
            column.array[start : start + 1024].isna().any() for start in range(0, len(column), 1024)
        ):
            # Missing integers use floats. Missing booleans use objects.
            dtype = np.dtype(object if dtype.kind == "b" else np.float64)

    for start in range(0, len(data), 128):
        chunk = data.iloc[start : start + 128].to_numpy(dtype=dtype)
        for i in range(len(chunk)):
            yield chunk[i]


class PandasStreamer(BaseStreamer):
    """Stream DataFrame rows as NumPy arrays.

    If shuffle is False, convert one row at a time.
    Keep the row order and omit the index.

    Args:
        shuffle (bool): Whether to shuffle the data initially (Default=False).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the shuffling. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self, shuffle: bool = False, random_state: int | np.random.RandomState | None = None
    ) -> None:
        super().__init__(shuffle=shuffle)

        self.array_iterator = ArrayStreamer(shuffle=shuffle, random_state=random_state)

    @overload
    def iter(self, X: pd.DataFrame, y: None = None) -> Iterator[np.ndarray]: ...

    @overload
    def iter(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series
    ) -> Iterator[tuple[np.ndarray, Any]]: ...

    def iter(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None = None
    ) -> Iterator[np.ndarray | tuple[np.ndarray, Any]]:
        """Yield feature rows and optional labels in positional order.

        If shuffle is True, yield the rows in shuffled order.
        Series labels give scalars. DataFrame labels give one-dimensional arrays.

        Args:
            X (pd.DataFrame): The features of the instances to iterate.
            y (pd.DataFrame, pd.Series or None): If not None, the labels to iterate in the same order (Default=None).
        """
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        if not self.shuffle:
            if y is None:
                yield from _iter_rows(X)
            else:
                yield from zip(_iter_rows(X), _iter_rows(y), strict=True)
            return

        if y is None:
            yield from self.array_iterator.iter(X.to_numpy())
        else:
            yield from self.array_iterator.iter(X.to_numpy(), y.to_numpy())
