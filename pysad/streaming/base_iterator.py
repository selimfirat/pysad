from abc import abstractmethod


class BaseIterator:

    def __init__(self, shuffle=False, **kwargs):
        self.shuffle = shuffle
        self.kwargs = kwargs

    @abstractmethod
    def iter(self, X, y=None):

        pass
