from abc import abstractmethod

from streaming.array_iterator import ArrayIterator


class BaseModel:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def fit_partial(self, X, y=None):
        pass

    @abstractmethod
    def score_partial(self, X):
        pass

    def fit(self, X, y=None):
        for xi, yi in self._iterate(X, y):
            self.fit_partial(xi, yi)

        return self

    def score(self, X):
        for xi, _ in self._iterate(X):
            self.score_partial(xi)

        return self

    def _iterate(self, X, y=None):
        iterator = ArrayIterator(shuffle=False)

        if y is None:
            for xi in iterator.iter(X):
                yield xi, None
        else:
            for xi, yi in iterator.iter(X, y):
                yield xi, yi
