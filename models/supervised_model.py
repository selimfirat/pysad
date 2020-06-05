from abc import abstractmethod


class SpervisedModel:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def fit_partial(self, x, y):
        pass

    @abstractmethod
    def predict_partial(self, x, y):
        pass
