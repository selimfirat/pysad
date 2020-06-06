from abc import abstractproperty, abstractmethod

from sklearn.metrics import roc_auc_score
from evaluation.base_metric import BaseMetric

class BaseSKLearnMetric(BaseMetric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.y_true = []
        self.y_pred = []

    def update(self, y_true, y_pred):

        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def get(self):

        score = self.score_method(self.y_true, self.y_pred)

        return score

    @property
    @abstractmethod
    def score_method(self):
        pass
