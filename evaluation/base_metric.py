from abc import abstractmethod


class BaseMetric:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.score = None

    @abstractmethod
    def update(self, y_true, y_pred):
        pass

    @abstractmethod
    def get(self):

        return self.score
