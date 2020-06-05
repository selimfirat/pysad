import mlflow
from sklearn.metrics import roc_auc_score

from evaluation.base_evaluator import BaseEvaluator


class AUROC(BaseEvaluator):

