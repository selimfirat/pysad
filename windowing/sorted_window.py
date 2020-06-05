from windowing.window import Window


class SortedWindow(Window):

    def update(self, num):
        heappush
        if len(self.window) > self.window_size:
            self.window = self.window[1:]
