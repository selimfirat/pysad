from abc import abstractmethod, ABCMeta
from pysad.evaluation.base_metric import BaseMetric


class BaseSKLearnMetric(BaseMetric, metaclass=ABCMeta):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.y_true = []
        self.y_pred = []

    def update(self, y_true, y_pred):

        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def get(self):

        score = self.evaluate(self.y_true, self.y_pred)

        return score

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        pass
