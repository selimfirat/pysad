from sklearn.metrics import roc_auc_score
from evaluation.base_sklearn_metric import BaseSKLearnMetric


class AUROCMetric(BaseSKLearnMetric):

    score_method = roc_auc_score
