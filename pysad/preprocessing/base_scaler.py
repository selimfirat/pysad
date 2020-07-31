from abc import abstractmethod


class BaseScaler:

    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()

    def fit_transform_partial(self, X):
        self.fit_partial(X)

        return self.transform_partial(X)

    @abstractmethod
    def fit_partial(self, X):
        pass

    @abstractmethod
    def transform_partial(self, X):
        pass
