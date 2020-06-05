import mlflow
from sklearn.metrics import roc_auc_score

from evaluation.base_metric import BaseMetric


class AUROC(BaseMetric):
    def update(self, y_true, y_pred):
        self.score =


    def get(self):

        return self.score
