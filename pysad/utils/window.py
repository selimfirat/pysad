from typing import Any

import numpy as np


class Window:
    """Window to limit the instances in list and keep the size fixed when full.

    The window owns its items: NumPy array items are copied on update, so a caller
    may overwrite or reuse an array after updating without changing the retained items.

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
        self.window: list[Any] = []

    def update(self, num: Any) -> None:
        """Adds new item to the window. Removes the tail if size exceeds the self.window_size.

        Args:
            num (float): item to be added to the window.
        """
        if isinstance(num, np.ndarray):
            num = num.copy()
        self.window.append(num)
        if self.window_size is not None and len(self.window) > self.window_size:
            self.window = self.window[1:]

    def get(self) -> list[Any]:
        """Method to obtain the window list.

        Returns:
            list: The items in the window.
        """
        return self.window


class UnlimitedWindow(Window):
    """Unlimited window implemented for convenience. This class only provides a list with unlimited size. Note that this does not fit to the memory for streaming data."""

    def __init__(self) -> None:
        super().__init__(None)

    def update(self, num: Any) -> None:
        """Adds new item to the window. Removes the tail if size exceeds the self.window_size.

        Args:
            num (float): item to be added to the window.
        """
        if isinstance(num, np.ndarray):
            num = num.copy()
        self.window.append(num)
