from abc import ABC, abstractmethod


class BasePostProcessor(ABC):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def fit_partial(self, score):
        pass

    @abstractmethod
    def transform_partial(self, score):
        pass

    def fit_transform_partial(self, score):
        self.fit_partial(score)

        return self.transform_partial(score)
