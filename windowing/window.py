

class Window:

    def __init__(self, window_size, **kwargs):

        self.kwargs = kwargs
        self.window_size = window_size
        self.window = []

    def update(self, num):
        self.window.append(num)
        if len(self.window) > self.window_size:
            self.window = self.window[1:]
