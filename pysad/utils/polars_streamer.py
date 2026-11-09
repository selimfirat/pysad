from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from pysad.core.base_streamer import BaseStreamer
from pysad.utils.array_streamer import ArrayStreamer

if TYPE_CHECKING:
    import polars as pl


def _iter_rows(data: pl.DataFrame | pl.Series) -> Iterator[Any]:
    import polars as pl

    if isinstance(data, pl.Series):
        for row in _iter_rows(data.to_frame()):
            yield row[0]
        return

    # Keep null-based dtype promotion the same in every buffer.
    sample = data.head(1).with_columns(column.clear(1) for column in data if column.null_count())
    for buffer in data.iter_slices(n_rows=512):
        # NumPy conversion preserves nanoseconds that Python row values lose.
        yield from buffer.vstack(sample).to_numpy()[:-1]


class PolarsStreamer(BaseStreamer):
    """Stream Polars DataFrame rows as NumPy arrays.

    If shuffle is False, convert fixed-size row buffers.
    Keep row order, NumPy dtypes, and timestamp precision.

    Args:
        shuffle (bool): Shuffle rows before iteration (Default=False).
        random_state (int, np.random.RandomState or None): Seed or generator for shuffling (Default=None).
            None draws from NumPy's global random state. `pysad.utils.fix_seed` seeds that state.
    """

    def __init__(
        self, shuffle: bool = False, random_state: int | np.random.RandomState | None = None
    ) -> None:
        super().__init__(shuffle=shuffle)

        self.array_iterator = ArrayStreamer(shuffle=shuffle, random_state=random_state)

    @overload
    def iter(self, X: pl.DataFrame, y: None = None) -> Iterator[np.ndarray]: ...

    @overload
    def iter(
        self, X: pl.DataFrame, y: pl.DataFrame | pl.Series
    ) -> Iterator[tuple[np.ndarray, Any]]: ...

    def iter(
        self, X: pl.DataFrame, y: pl.DataFrame | pl.Series | None = None
    ) -> Iterator[np.ndarray | tuple[np.ndarray, Any]]:
        """Yield feature rows and optional labels in positional order.

        If shuffle is True, yield the rows in shuffled order.
        Series labels give scalars. DataFrame labels give one-dimensional arrays.

        Args:
            X (pl.DataFrame): Feature rows to stream.
            y (pl.DataFrame, pl.Series or None): Labels in the same row order (Default=None).
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
