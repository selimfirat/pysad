

class Window:
    """Window to limit the instances in list and keep the size fixed when full.

    Args:
        window_size (int): The size of the window.
    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.window = []

    def update(self, num):
        """Adds new item to the window. Removes the tail if size exceeds the self.window_size.

        Args:
            num (float): item to be added to the window.
        """
        self.window.append(num)
        if len(self.window) > self.window_size:
            self.window = self.window[1:]

    def get(self):
        """Method to obtain the window list.

        Returns:
            window (list): List containing the items in window
        """
        return self.window


class UnlimitedWindow(Window):
    """Unlimited window implemented for convenience. This class only provides a list with unlimited size. Note that this does not fit to the memory for streaming data.
    """

    def __init__(self):
        super().__init__(None)

    def update(self, num):
        """Adds new item to the window. Removes the tail if size exceeds the self.window_size.

        Args:
            num (float): item to be added to the window.
        """
        self.window.append(num)
