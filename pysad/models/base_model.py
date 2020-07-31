from abc import abstractmethod, ABC
from pysad.utils import _iterate


class BaseModel(ABC):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def fit_partial(self, X, y=None):
        pass

    @abstractmethod
    def score_partial(self, X):
        pass

    def fit(self, X, y=None):
        for xi, yi in _iterate(X, y):
            self.fit_partial(xi, yi)

        return self

    def score(self, X):
        for xi, _ in _iterate(X):
            yield self.score_partial(xi)
