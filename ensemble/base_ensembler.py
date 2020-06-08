from abc import ABC, abstractmethod

from streaming.array_iterator import ArrayIterator


class BaseEnsembler(ABC):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def transform_partial(self, scores):
        pass

    def transform(self, scores):
        for xi, _ in self._iterate(scores):
            yield self.transform_partial(scores)

    def _iterate(self, X):
        iterator = ArrayIterator(shuffle=False)
        for xi in iterator.iter(X):
            yield xi, None
