from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


def _kth_abs_dev(
    sub: Sequence[float] | NDArray[np.float64], center: float, h: int, k: int
) -> float:
    """Returns the k-th smallest value (0-indexed) of ``|sub - center|`` when ``sub`` is sorted, in O(log m) time."""
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


def _median_and_mad_sorted(sub: Sequence[float] | NDArray[np.float64]) -> tuple[float, float]:
    """Returns the median and the median absolute deviation of a non-empty sorted 1-D sequence ``sub`` in O(log m) time."""
    m = len(sub)
    h = m >> 1
    center = float(sub[h]) if (m & 1) else 0.5 * (float(sub[h - 1]) + float(sub[h]))
    if m & 1:
        mad = _kth_abs_dev(sub, center, h, h)
    else:
        mad = 0.5 * (_kth_abs_dev(sub, center, h, h - 1) + _kth_abs_dev(sub, center, h, h))
    return center, mad


class RingBuffer:
    """Fixed-capacity circular buffer of ``np.float64`` items.

    The first appended item fixes the item shape. A scalar item makes a buffer with shape
    ``(window_size,)``, and a 1-D vector item makes a buffer with shape
    ``(window_size, num_features)``. Later items must keep the shape of the first item.

    Args:
        window_size (int): Maximum number of items that the buffer retains. Must be at least 1.

    Raises:
        ValueError: If ``window_size`` is less than 1.
    """

    def __init__(self, window_size: int) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        self.window_size = window_size
        self.points: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self.count = 0
        self.next_slot = 0
        self._item_shape: tuple[int, ...] | None = None

    def append(self, item: float | NDArray[Any]) -> int:
        """Writes ``item`` into the next free slot and returns the slot index.

        The next append writes to the following slot. At the end of the buffer, the slot index
        wraps to 0 and the oldest item expires.

        Args:
            item (float or np.ndarray): Scalar value or 1-D feature vector to store. Its shape must
                match the item shape of the first stored item.

        Returns:
            int: Slot index that received the item.

        Raises:
            ValueError: If the item shape differs from the item shape of the first stored item.
        """
        arr = np.asarray(item, dtype=np.float64)
        if self.count == 0 and self.points.size == 0:
            if arr.ndim == 0:
                self._item_shape = ()
                self.points = np.empty((self.window_size,), dtype=np.float64)
            else:
                self._item_shape = arr.shape
                self.points = np.empty((self.window_size,) + arr.shape, dtype=np.float64)
        elif self._item_shape != arr.shape:
            raise ValueError(f"item must have shape {self._item_shape}, got {arr.shape}.")
        slot = self.next_slot
        self.points[slot] = arr
        self.next_slot = (slot + 1) % self.window_size
        if self.count < self.window_size:
            self.count += 1
        return slot

    def extend(self, items: Sequence[float | NDArray[Any]] | NDArray[Any]) -> None:
        """Appends multiple items to the buffer in a vectorized manner.

        Args:
            items: Sequence of items to append, or a 2-D array of shape (n_items, ...).
                Each item must have the same shape as the first item appended to the buffer.
        """
        if len(items) == 0:
            return
        arr = np.asarray(items, dtype=np.float64)

        # Handle scalar items: if the buffer expects scalars (shape ()), a 1-D array of
        # shape (n,) should be treated as n scalar items. We keep it as 1-D for assignment.
        is_scalar_batch = self._item_shape == () and arr.ndim == 1

        if self.count == 0 and self.points.size == 0:
            if arr.ndim == 1 or is_scalar_batch:
                self._item_shape = ()
                self.points = np.empty((self.window_size,), dtype=np.float64)
            else:
                self._item_shape = arr.shape[1:]
                self.points = np.empty((self.window_size,) + arr.shape[1:], dtype=np.float64)
        elif self._item_shape != (arr.shape[1:] if arr.ndim > 1 else ()):
            raise ValueError(
                f"items must have shape {self._item_shape}, got {arr.shape[1:] if arr.ndim > 1 else ()}."
            )

        n = arr.shape[0]
        if n >= self.window_size:
            # Batch larger than window: keep only the last window_size items
            if is_scalar_batch:
                self.points[:] = arr[-self.window_size :]
            else:
                self.points[:] = arr[-self.window_size :]
            self.next_slot = 0
            self.count = self.window_size
            return

        # Vectorized write using modular arithmetic
        end_slot = self.next_slot + n
        if end_slot <= self.window_size:
            # No wrap
            self.points[self.next_slot : end_slot] = arr
        else:
            # Wrap around
            first_part = self.window_size - self.next_slot
            self.points[self.next_slot :] = arr[:first_part]
            self.points[: end_slot - self.window_size] = arr[first_part:]
        self.next_slot = end_slot % self.window_size
        self.count = min(self.count + n, self.window_size)

    def active_slices(self) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, int]:
        """Returns the stored items that remain after the oldest item of a full buffer expires.

        The method returns views of the stored array, not copies. When the buffer is not full, it
        returns ``(items, None, num_items)``, where ``items`` holds the stored items oldest first.
        When the buffer is full, the next append expires the oldest item, so the method returns
        the remaining items in two slices: ``second`` holds the older items, and ``first`` holds
        the newer items.

        Returns:
            tuple: ``(first_slice, second_slice_or_None, active_count)``, with
                ``second_slice`` equal to None before the buffer is full.
        """
        if self.count < self.window_size:
            return self.points[: self.count], None, self.count
        slot = self.next_slot
        return self.points[:slot], self.points[slot + 1 :], self.window_size - 1

    def window_slices(self) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """Returns the window items in chronological order as one or two slices (views, no copy).

        Unlike ``window_view``, this method never concatenates. When the buffer has not wrapped
        or ``next_slot == 0``, returns ``(items, None)``. Otherwise returns ``(newer, older)``
        where ``newer = points[next_slot:]`` and ``older = points[:next_slot]``.

        Returns:
            tuple: ``(first_slice, second_slice_or_None)``. The chronological order is
                ``np.concatenate((first, second))`` if second is not None, else ``first``.
        """
        if self.count == 0:
            return self.points[:0], None
        if self.count < self.window_size or self.next_slot == 0:
            return self.points[: self.count], None
        return self.points[self.next_slot :], self.points[: self.next_slot]

    def window_view(self) -> NDArray[np.float64]:
        """Returns the window items in chronological order (oldest first) as a view when possible.

        If the buffer has not wrapped or ``next_slot == 0``, returns a zero-copy view of the
        underlying storage. Otherwise, returns a concatenated array (which copies).

        Returns:
            np.float64 array: Stored items in chronological order, shape ``(count, ...)``.
        """
        if self.count == 0:
            return self.points[:0]
        if self.count < self.window_size or self.next_slot == 0:
            return self.points[: self.count]
        # Use np.roll for a single allocation instead of concatenate
        return np.roll(self.points[: self.window_size], -self.next_slot, axis=0)

    def oldest(self) -> float | NDArray[np.float64] | None:
        """Returns the oldest item in the buffer, or ``None`` if empty.

        This is the item that will be evicted on the next append when the buffer is full.

        Returns:
            The oldest item, or ``None`` if the buffer is empty.
        """
        if self.count == 0:
            return None
        if self.count < self.window_size:
            item = self.points[0]
            if item.ndim == 0:
                return float(item)
            return cast(NDArray[np.float64], item)
        item = self.points[self.next_slot]
        if item.ndim == 0:
            return float(item)
        return cast(NDArray[np.float64], item)

    def clear(self) -> None:
        """Removes all items from the buffer."""
        self.count = 0
        self.next_slot = 0

    def to_array(self) -> NDArray[np.float64]:
        """Returns a contiguous oldest-first copy of the items currently in the buffer.

        Returns:
            np.float64 array: Stored items in chronological order. The array is empty when the
                buffer holds no item.
        """
        return self.window_view().copy()

    def __len__(self) -> int:
        """Returns the number of items currently stored in the buffer."""
        return self.count


class Window:
    """Keeps the last ``window_size`` items of a stream.

    The window holds at most ``window_size`` items, oldest first. Adding an item beyond the limit
    drops the oldest item in constant time, independent of the window size.

    This implementation uses a ring buffer for better cache locality and vectorized access
    compared to a Python list.

    Args:
        window_size (int): The size of the window. Must be at least 1.

    Raises:
        ValueError: If ``window_size`` is less than 1.
    """

    def __init__(self, window_size: int) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")

        self.window_size = window_size
        self._buffer = RingBuffer(window_size)

    def update(self, item: Any) -> None:
        """Adds an item. If the window then holds more than ``window_size`` items, removes the oldest one.

        Adding an item takes constant time, independent of the window size.

        Args:
            item (Any): Item to be added to the window. Must be convertible to float.
        """
        self._buffer.append(float(item))

    def update_batch(self, items: Sequence[float] | NDArray[Any]) -> None:
        """Adds multiple items in a vectorized manner.

        Args:
            items: Sequence of items to append, or a 1-D array of floats.
        """
        arr = np.asarray(items, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("items must be a 1-D sequence or array of floats.")
        self._buffer.extend(arr)

    def get(self) -> list[float]:
        """Returns a copy of the window items, oldest first.

        Returns:
            list[float]: The items in the window, oldest first.
        """
        return cast(list[float], self._buffer.window_view().tolist())

    def get_array(self) -> NDArray[np.float64]:
        """Returns a copy of the window items as a NumPy array, oldest first.

        This avoids the Python list conversion of ``get()`` for better performance.

        Returns:
            np.float64 array: The items in the window, oldest first.
        """
        return self._buffer.window_view().copy()

    def __len__(self) -> int:
        """Returns the number of items currently in the window."""
        return len(self._buffer)

    def __iter__(self) -> Iterator[float]:
        """Iterates over the items in the window, oldest first."""
        return iter(cast(list[float], self._buffer.window_view().tolist()))
