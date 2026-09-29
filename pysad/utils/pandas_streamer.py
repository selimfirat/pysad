from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from pysad.core.base_streamer import BaseStreamer
from pysad.utils.array_streamer import ArrayStreamer

if TYPE_CHECKING:
    import pandas as pd


class PandasStreamer(BaseStreamer):
    """Simulator class to iterate dataframe(s).

    Args:
        shuffle (bool): Whether shuffle the data initially (Default=False).
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
        """Iterates pandas dataframes of of features and possibly labels.

        Args:
            X: Pandas Dataframe for features.
            y: Pandas dataframe for labels.
        """
        if y is None:
            for x in self.array_iterator.iter(X.to_numpy()):
                yield x
        else:
            if len(X) != len(y):
                raise ValueError("X and y must have the same length.")

            for x, yr in self.array_iterator.iter(X.to_numpy(), y.to_numpy()):
                yield x, yr
