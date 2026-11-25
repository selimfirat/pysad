"""Streaming statistics that wrap the bounded-memory sketches in :mod:`pysad.utils`."""

from __future__ import annotations

from typing import Any

import numpy as np

from pysad.core.base_statistic import BaseStatistic
from pysad.utils import CountMinSketch, HyperLogLog


class CountMinSketchStatistic(BaseStatistic):
    """The estimated frequency of the most recent stream value, tracked with a Count-Min Sketch.

    Wraps :class:`pysad.utils.CountMinSketch` so it can be used like any other streaming
    statistic. The sketch keeps a bounded ``depth`` by ``width`` array of counters for any stream
    length and never stores the items. Adding a value increments one counter per row. The
    frequency estimate of a value is the smallest of its counters, so hash collisions can only
    overestimate the true frequency and never underestimate it.

    With ``width = ceil(e / epsilon)`` and ``depth = ceil(ln(1 / delta))``, the estimate exceeds
    the true frequency by more than ``epsilon * total`` with probability at most ``delta``
    (Theorem 1 of :cite:`cormode2005countmin`). Two statistics merge by folding their sketches
    together when they share ``width``, ``depth`` and hash functions, which holds when both are
    built with the same ``random_state``.

    Deviations from the paper: the integer key of an item is ``blake2b`` over its ``repr``, as in
    :class:`pysad.utils.CountMinSketch`. Equal items with different representations, such as
    ``1`` and ``True``, count as different items. The sketch accepts only non-negative counts, so
    this statistic does not support removing values; build a fresh statistic per window instead.

    Args:
        width (int): Number of counters per row (Default=2719).
        depth (int): Number of rows, each with its own hash function (Default=5).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the
            hash functions. None draws from NumPy's global random state (Default=None).

    Attributes:
        width (int): Number of counters per row.
        depth (int): Number of rows.
    """

    _removable = False

    def __init__(
        self,
        width: int = 2719,
        depth: int = 5,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self._sketch = CountMinSketch(width=width, depth=depth, random_state=random_state)
        self._last_value: Any = None

    def update(self, num: Any) -> CountMinSketchStatistic:
        """Updates the statistic with one value.

        Args:
            num (Any): The incoming hashable value. The sketch keeps no reference to it.

        Returns:
            CountMinSketchStatistic: self.
        """
        self._sketch.update(num)
        self._last_value = num
        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The estimated frequency of the most recently updated value, or 0.0 before the
            first update.
        """
        if self._last_value is None:
            return 0.0
        return float(self._sketch.estimate(self._last_value))

    def merge(self, other: CountMinSketchStatistic) -> CountMinSketchStatistic:
        """Folds another statistic's sketch into this one.

        Args:
            other (CountMinSketchStatistic): The statistic to fold in. It stays unchanged.

        Returns:
            CountMinSketchStatistic: self.

        Raises:
            TypeError: If ``other`` is not a CountMinSketchStatistic.
            ValueError: If the sketches have different widths, depths, or hash functions.
        """
        if not isinstance(other, CountMinSketchStatistic):
            raise TypeError(f"Cannot merge {type(other).__name__} into CountMinSketchStatistic.")
        self._sketch.merge(other._sketch)
        return self


class HyperLogLogStatistic(BaseStatistic):
    """The estimated number of distinct stream values, tracked with HyperLogLog.

    Wraps :class:`pysad.utils.HyperLogLog` so it can be used like any other streaming statistic.
    HyperLogLog keeps one byte per register and never stores the items. It hashes each value to
    64 bits, uses the first ``precision`` bits to select a register, and records the largest rank
    seen in each register. The estimator combines the registers with the constants from the
    paper, using linear counting for small streams and a large-range correction near the hash
    range (:cite:`flajolet2007hyperloglog`).

    Two statistics with the same ``precision`` merge by keeping each register's larger rank,
    which estimates the cardinality of the union without revisiting the values.

    Deviations from the paper: each value's fingerprint is ``blake2b`` over its ``repr``, as in
    :class:`pysad.utils.HyperLogLog`. Estimates reproduce across runs and machines only for items
    with stable representations. Equal items with different representations, such as ``1`` and
    ``True``, count as distinct values. HyperLogLog has no removal operation, so this statistic
    does not support removing values; build a fresh statistic per window instead.

    Args:
        precision (int): Number of hash bits that select a register (Default=14). The supported
            range is 4 to 18, inclusive. The registers use ``2 ** precision`` bytes.

    Attributes:
        precision (int): Number of hash bits that select a register.
    """

    _removable = False

    def __init__(self, precision: int = 14) -> None:
        self._sketch = HyperLogLog(precision=precision)

    def update(self, num: Any) -> HyperLogLogStatistic:
        """Updates the statistic with one value.

        Args:
            num (Any): The incoming hashable value. The sketch keeps no reference to it.

        Returns:
            HyperLogLogStatistic: self.
        """
        self._sketch.update(num)
        return self

    def get(self) -> float:
        """Method to obtain the tracked statistic.

        Returns:
            float: The current cardinality estimate of the stream.
        """
        return float(self._sketch.estimate())

    def merge(self, other: HyperLogLogStatistic) -> HyperLogLogStatistic:
        """Folds another statistic's sketch into this one.

        Args:
            other (HyperLogLogStatistic): The statistic to fold in. It stays unchanged.

        Returns:
            HyperLogLogStatistic: self.

        Raises:
            TypeError: If ``other`` is not a HyperLogLogStatistic.
            ValueError: If the sketches use different precisions.
        """
        if not isinstance(other, HyperLogLogStatistic):
            raise TypeError(f"Cannot merge {type(other).__name__} into HyperLogLogStatistic.")
        self._sketch.merge(other._sketch)
        return self
