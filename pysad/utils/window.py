from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np


def _kth_abs_dev(sub: Sequence[float] | np.ndarray, center: float, h: int, k: int) -> float:
    """Returns the 0-indexed k-th smallest element of ``|sub - center|`` for sorted ``sub`` in O(log m)."""
    left_len = h
    right_len = len(sub) - h
    low = max(0, k - right_len)
    high = min(k, left_len)
    while low < high:
        i = (low + high) >> 1
        j = k - i
        if float(sub[h + j - 1]) - center > center - float(sub[h - 1 - i]):
            low = i + 1
        else:
            high = i
    i = low
    j = k - i
    if i == left_len:
        return float(sub[h + j]) - center
    if j == right_len:
        return center - float(sub[h - 1 - i])
    a_val = center - float(sub[h - 1 - i])
    b_val = float(sub[h + j]) - center
    return a_val if a_val <= b_val else b_val


def _median_and_mad_sorted(sub: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """Returns ``(median, MAD)`` of a non-empty sorted 1-D sequence ``sub`` in O(log m) time."""
    m = len(sub)
    h = m >> 1
    center = float(sub[h]) if (m & 1) else 0.5 * (float(sub[h - 1]) + float(sub[h]))
    if m & 1:
        mad = _kth_abs_dev(sub, center, h, h)
    else:
        mad = 0.5 * (_kth_abs_dev(sub, center, h, h - 1) + _kth_abs_dev(sub, center, h, h))
    return center, mad


class RingBuffer:
    """Fixed-capacity contiguous ``np.float64`` circular buffer for 1-D scalars or 2-D vectors.

    The underlying array is allocated lazily on the first :meth:`append` call so the buffer matches
    either scalar values ``(window_size,)`` or feature vectors ``(window_size, num_features)``
    without upfront shape metadata.

    Args:
        window_size (int): Maximum number of items retained in the buffer. Must be at least 1.

    Raises:
        ValueError: If ``window_size`` is less than 1.
    """

    def __init__(self, window_size: int) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        self.window_size = window_size
        self.points: np.ndarray = np.empty(0, dtype=np.float64)
        self.count = 0
        self.next_slot = 0

    def append(self, item: float | np.ndarray) -> int:
        """Writes ``item`` into the next slot, advances the ring, and returns the written slot index.

        Args:
            item (float or np.ndarray): Scalar value or 1-D feature vector to store.

        Returns:
            int: Slot index written in ``points``.
        """
        if self.count == 0 and self.points.size == 0:
            arr = np.asarray(item, dtype=np.float64)
            shape = (self.window_size,) if arr.ndim == 0 else (self.window_size, arr.shape[0])
            self.points = np.empty(shape, dtype=np.float64)
        slot = self.next_slot
        self.points[slot] = item
        self.next_slot = (slot + 1) % self.window_size
        if self.count < self.window_size:
            self.count += 1
        return slot

    def active_slices(self) -> tuple[np.ndarray, np.ndarray | None, int]:
        """Returns zero-copy slice(s) of the window after the oldest slot of a full window expires.

        Returns:
            tuple: ``(first_slice, second_slice_or_None, active_count)``.
        """
        if self.count < self.window_size:
            return self.points[: self.count], None, self.count
        slot = self.next_slot
        return self.points[:slot], self.points[slot + 1 :], self.window_size - 1

    def to_array(self) -> np.ndarray:
        """Returns a contiguous oldest-first copy of the items currently in the buffer.

        Returns:
            np.float64 array: Stored items in chronological order.
        """
        if self.count == 0:
            return self.points[:0].copy()
        if self.count < self.window_size or self.next_slot == 0:
            return self.points[: self.count].copy()
        return np.concatenate((self.points[self.next_slot :], self.points[: self.next_slot]))

    def __len__(self) -> int:
        """Returns the number of items currently stored in the buffer."""
        return self.count


class Window:
    """Window to limit the instances in a list and keep the size fixed when full.

    The window holds at most ``window_size`` items, oldest first. Adding an item beyond the limit
    drops the oldest item in constant time, independent of the window size.

    Args:
        window_size (int): The size of the window. Must be at least 1.

    Raises:
        ValueError: If window_size is less than 1.
    """

    def __init__(self, window_size: int | None) -> None:
        # UnlimitedWindow uses None as its internal size sentinel.
        if window_size is not None and window_size < 1:
            raise ValueError("window_size must be a positive integer.")

        self.window_size = window_size
        self.window: deque[Any] = deque(maxlen=window_size)

    def update(self, num: Any) -> None:
        """Adds new item to the window. Removes the oldest item if size exceeds window_size.

        Adding an item takes constant time, independent of the window size.

        Args:
            num (float): Item to be added to the window.
        """
        self.window.append(num)

    def get(self) -> list[Any]:
        """Method to obtain the window items.

        Returns:
            list: The items in the window, oldest first.
        """
        return list(self.window)

    def __len__(self) -> int:
        """Returns the number of items currently in the window."""
        return len(self.window)

    def __iter__(self) -> Iterator[Any]:
        """Iterates over the items in the window, oldest first."""
        return iter(self.window)


class UnlimitedWindow(Window):
    """Unlimited window implemented for convenience. This class only provides a list with unlimited size. Note that this does not fit to the memory for streaming data."""

    def __init__(self) -> None:
        super().__init__(None)

    def update(self, num: Any) -> None:
        """Adds new item to the window. The window never drops items.

        Args:
            num (float): Item to be added to the window.
        """
        self.window.append(num)
