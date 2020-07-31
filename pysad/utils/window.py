

class Window:

    def __init__(self, window_size):
        """Window to limit the instances in list and keep the size fixed when full.

        Args:
            window_size: The size of the window.
        """
        self.window_size = window_size
        self.window = []

    def update(self, num):
        """Adds new item to the window. Removes the tail if size exceeds the self.window_size.

        Args:
            num: item to be added to the window.
        """
        self.window.append(num)
        if len(self.window) > self.window_size:
            self.window = self.window[1:]

    def get(self):
        """Method to obtain the window list.

        Returns:
            window: list
                list containing the items in window

        """
        return self.window
