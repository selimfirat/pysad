from abc import abstractmethod


class UnsupervisedModel:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def fit_partial(self, X):
        pass

    @abstractmethod
    def predict_partial(self, X):
        pass
