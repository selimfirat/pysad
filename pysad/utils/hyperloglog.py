"""Estimate the number of distinct items in a stream with bounded memory."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from typing import Any

_HASH_BITS = 64
"""Number of bits in each fingerprint."""

_ALPHA_BY_PRECISION = {4: 0.673, 5: 0.697, 6: 0.709}
"""Figure 3 constants for precisions without a closed form."""


class HyperLogLog:
    """Estimate the cardinality of a stream of hashable items in bounded memory :cite:`flajolet2007hyperloglog`.

    HyperLogLog keeps one byte per register and never stores the items.
    The registers use ``2 ** precision`` bytes for any stream length.
    The sketch hashes each item to 64 bits.
    The first ``precision`` bits select a register.
    The leading zeros in the remaining bits determine the rank.
    Each register keeps the largest rank it sees.
    The estimator combines these ranks with the constants from the paper.
    For small counts, it uses linear counting.
    Near the hash range, it applies the paper's large-range correction.

    Two sketches with the same ``precision`` merge by keeping each bucket's
    larger rank. This estimates the union without revisiting the items.

    Deviations from the paper: the sketch uses ``blake2b`` over each item's
    ``repr`` instead of a random universal hash.
    Estimates reproduce across runs and machines only for items with stable
    representations. Frozenset order can change across runs.
    Default object representations include memory addresses.
    Equal items with different representations, such as ``1`` and ``True``,
    count as distinct values.
    A custom ``__repr__`` that omits distinguishing fields can undercount.

    Args:
        precision (int): Number of hash bits that select a register (Default=14).
            The supported range is 4 to 18, inclusive.
            The registers use ``2 ** precision`` bytes.
            Higher values reduce the standard error to about
            ``1.04 / sqrt(2 ** precision)``.
            The default uses 16 KiB with about 0.8% standard error.

    Raises:
        ValueError: If ``precision`` falls outside the supported range of 4 to 18.
    """

    def __init__(self, precision: int = 14) -> None:
        if not 4 <= precision <= 18:
            raise ValueError(f"precision must be between 4 and 18, got {precision}")
        self.precision = precision
        self._registers = bytearray(1 << precision)
        self._width = _HASH_BITS - precision

    @property
    def num_registers(self) -> int:
        """Number of registers, which is also the memory of the sketch in bytes.

        Returns:
            int: ``2 ** precision``.
        """
        return len(self._registers)

    def update(self, item: Any) -> None:
        """Add one item to the sketch.

        Args:
            item (Any): A hashable item. The sketch keeps no reference to it.
        """
        fingerprint = _fingerprint(item)
        index = fingerprint >> self._width
        rest = fingerprint & ((1 << self._width) - 1)
        rank = self._width - rest.bit_length() + 1
        if rank > self._registers[index]:
            self._registers[index] = rank

    def update_many(self, items: Iterable[Any]) -> None:
        """Add every item of an iterable to the sketch.

        Args:
            items (Iterable[Any]): Hashable items. The sketch reads the iterable once.
        """
        for item in items:
            self.update(item)

    def merge(self, other: HyperLogLog) -> None:
        """Fold another sketch of the same precision into this one.

        The result estimates the cardinality of the union of both streams.

        Args:
            other (HyperLogLog): The sketch to fold in. This operation does not change it.

        Raises:
            ValueError: If the other sketch uses a different ``precision``.
        """
        if other.precision != self.precision:
            raise ValueError(
                f"cannot merge precision {other.precision} into precision {self.precision}"
            )
        registers = self._registers
        for index, rank in enumerate(other._registers):
            if rank > registers[index]:
                registers[index] = rank

    def estimate(self) -> float:
        """Estimate the number of distinct items in the stream.

        Returns:
            float: The estimate, with the small-range and large-range corrections of the paper.
        """
        registers = self._registers
        m = len(registers)
        alpha = _ALPHA_BY_PRECISION.get(self.precision, 0.7213 / (1 + 1.079 / m))
        harmonic = sum(math.ldexp(1.0, -rank) for rank in registers)
        raw = alpha * m * m / harmonic

        empty = registers.count(0)
        if raw <= 2.5 * m and empty:
            # Linear counting: exact enough while the stream is short.
            return m * math.log(m / empty)
        if raw > 2**64 / 30:
            # Estimates near 2**64 saturate, so undo the bias of the hash range.
            return -(2**64) * math.log(1 - raw / 2**64)
        return raw

    def __len__(self) -> int:
        """Return the estimate as a whole number.

        Returns:
            int: ``round(estimate())``.
        """
        return int(round(self.estimate()))

    def __repr__(self) -> str:
        """Return the precision and the current estimate.

        Returns:
            str: A short representation.
        """
        return f"{type(self).__name__}(precision={self.precision}, estimate={len(self)})"


def _fingerprint(item: Any) -> int:
    """Hash one item to the 64-bit fingerprint the registers consume.

    Args:
        item (Any): A hashable item.

    Returns:
        int: ``blake2b`` over the ``repr`` of the item, as an unsigned 64-bit integer.
    """
    digest = hashlib.blake2b(repr(item).encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


__all__ = ["HyperLogLog"]
