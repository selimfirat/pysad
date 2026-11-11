from collections import deque
from typing import Any


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
