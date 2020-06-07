from abc import ABC, abstractmethod

from streaming.array_iterator import ArrayIterator


class BaseProjector(ABC):

    def __init__(self, n_components, **kwargs):
        self.n_components = n_components

    @abstractmethod
    def fit_partial(self, X):
        pass

    @abstractmethod
    def transform_partial(self, X):
        pass

    def fit(self, X):
        for xi in self._iterate(X):
            self.fit_partial(xi)

        return self

    def fit_transform_partial(self, X):
        self.fit_partial(X)

        return self.transform_partial(X)

    def transform(self, X):
        for xi, _ in self._iterate(X):
            yield self.transform_partial(xi)

    def fit_transform(self, X):
        for xi, _ in self._iterate(X):
            yield self.fit_transform_partial(xi)

    def _iterate(self, X):
        iterator = ArrayIterator(shuffle=False)
        for xi in iterator.iter(X):
            yield xi, None
