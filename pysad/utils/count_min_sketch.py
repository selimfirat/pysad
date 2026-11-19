"""Estimate the frequencies of items in a stream with bounded memory."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.utils import check_random_state

_HASH_PRIME = (1 << 61) - 1
"""Mersenne prime that the pairwise-independent hash functions reduce modulo."""


class CountMinSketch:
    """Estimate the frequency of each hashable item of a stream in bounded memory :cite:`cormode2005countmin`.

    The sketch is a ``depth`` by ``width`` array of counters with one hash function per row.
    Adding ``count`` copies of an item adds ``count`` to the counter its hash selects in every row.
    The frequency estimate of an item is the smallest of its ``depth`` counters.
    Collisions can only add to a counter, so the estimate never falls below the true frequency.
    With ``width = ceil(e / epsilon)`` and ``depth = ceil(ln(1 / delta))``, the estimate exceeds the
    true frequency by more than ``epsilon * total`` with probability at most ``delta``, where
    ``total`` is the sum of all added counts (Theorem 1 of the paper).
    The memory is ``width * depth`` counters for any stream length.

    Two sketches merge by adding their counters. This needs the same ``width``, ``depth`` and hash
    functions, so build both with the same ``random_state`` seed.

    Row ``j`` maps an item to the counter ``((a_j * key + b_j) mod P) mod width``, with the prime
    ``P = 2**61 - 1`` and random integers ``1 <= a_j < P`` and ``0 <= b_j < P``, the
    pairwise-independent hash functions of the paper.

    Deviations from the paper: the integer ``key`` of an item is ``blake2b`` over its ``repr``,
    as in :class:`pysad.utils.HyperLogLog`.
    Equal items with different representations, such as ``1`` and ``True``, count as different
    items. Default object representations include memory addresses.
    The sketch takes only non-negative counts, the case the paper's point-query bound covers.

    Args:
        width (int): Number of counters per row (Default=2719, ``ceil(e / 0.001)``).
            The error bound ``epsilon`` is ``e / width``.
        depth (int): Number of rows, each with its own hash function (Default=5, ``ceil(ln(1 / 0.01))``).
            The failure probability ``delta`` of the bound is ``exp(-depth)``.
        random_state (int, np.random.RandomState or None): Seed or random number generator for the
            hash functions. None draws from NumPy's global random state, which
            `pysad.utils.fix_seed` seeds (Default=None).

    Raises:
        ValueError: If ``width`` or ``depth`` is less than 1.
    """

    def __init__(
        self,
        width: int = 2719,
        depth: int = 5,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if width < 1:
            raise ValueError(f"width must be a positive integer, got {width}")
        if depth < 1:
            raise ValueError(f"depth must be a positive integer, got {depth}")
        self.width = width
        self.depth = depth
        rng = check_random_state(random_state)
        self._hash_params = tuple(
            (
                int(rng.randint(1, _HASH_PRIME, dtype=np.int64)),
                int(rng.randint(0, _HASH_PRIME, dtype=np.int64)),
            )
            for _ in range(depth)
        )
        self._counts = np.zeros((depth, width), dtype=np.int64)
        self._rows = np.arange(depth)
        self.total = 0

    @classmethod
    def from_error_bound(
        cls,
        epsilon: float,
        delta: float,
        random_state: int | np.random.RandomState | None = None,
    ) -> CountMinSketch:
        """Build a sketch whose estimates exceed the true frequency by more than ``epsilon * total`` with probability at most ``delta``.

        Args:
            epsilon (float): Error bound as a fraction of the total count, in (0, 1).
            delta (float): Probability that an estimate exceeds the bound, in (0, 1).
            random_state (int, np.random.RandomState or None): Seed or random number generator
                for the hash functions (Default=None).

        Returns:
            CountMinSketch: A sketch with ``width = ceil(e / epsilon)`` and ``depth = ceil(ln(1 / delta))``.

        Raises:
            ValueError: If ``epsilon`` or ``delta`` is not in (0, 1).
        """
        if not 0 < epsilon < 1:
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
        if not 0 < delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        width = math.ceil(math.e / epsilon)
        depth = math.ceil(math.log(1 / delta))
        return cls(width=width, depth=depth, random_state=random_state)

    @property
    def epsilon(self) -> float:
        """Error bound of the estimates as a fraction of ``total``.

        Returns:
            float: ``e / width``.
        """
        return math.e / self.width

    def update(self, item: Any, count: int = 1) -> None:
        """Add ``count`` copies of an item to the sketch.

        Args:
            item (Any): A hashable item. The sketch keeps no reference to it.
            count (int): Number of copies to add (Default=1).

        Raises:
            ValueError: If ``count`` is negative.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        self._counts[self._rows, self._buckets(item)] += count
        self.total += count

    def update_many(self, items: Iterable[Any]) -> None:
        """Add one copy of every item of an iterable to the sketch.

        Args:
            items (Iterable[Any]): Hashable items. The sketch reads the iterable once.
        """
        for item in items:
            self.update(item)

    def estimate(self, item: Any) -> int:
        """Estimate how many copies of an item the sketch has seen.

        Args:
            item (Any): A hashable item.

        Returns:
            int: The smallest counter of the item over the rows. It is never below the true frequency.
        """
        return int(self._counts[self._rows, self._buckets(item)].min())

    def merge(self, other: CountMinSketch) -> None:
        """Fold another sketch into this one.

        The result estimates the frequencies of both streams together.

        Args:
            other (CountMinSketch): The sketch to fold in. This operation does not change it.

        Raises:
            ValueError: If the other sketch has a different ``width``, ``depth`` or hash functions.
        """
        if (other.width, other.depth) != (self.width, self.depth):
            raise ValueError(
                f"cannot merge a {other.depth}x{other.width} sketch into a {self.depth}x{self.width} sketch"
            )
        if other._hash_params != self._hash_params:
            raise ValueError(
                "cannot merge sketches with different hash functions; use the same random_state"
            )
        self._counts += other._counts
        self.total += other.total

    def __repr__(self) -> str:
        """Return the shape and the total count.

        Returns:
            str: A short representation.
        """
        return f"{type(self).__name__}(width={self.width}, depth={self.depth}, total={self.total})"

    def _buckets(self, item: Any) -> np.ndarray:
        """Map an item to its counter in every row.

        Args:
            item (Any): A hashable item.

        Returns:
            np.int64 array of shape (depth,): The counter index of the item in each row.
        """
        key = _key(item)
        return np.array(
            [((a * key + b) % _HASH_PRIME) % self.width for a, b in self._hash_params],
            dtype=np.int64,
        )


def _key(item: Any) -> int:
    """Hash one item to the integer key the hash functions consume.

    Args:
        item (Any): A hashable item.

    Returns:
        int: ``blake2b`` over the ``repr`` of the item, as an unsigned 64-bit integer.
    """
    digest = hashlib.blake2b(repr(item).encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


__all__ = ["CountMinSketch"]
