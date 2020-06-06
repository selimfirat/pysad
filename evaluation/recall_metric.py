from sklearn.metrics import recall_score
from evaluation.base_sklearn_metric import BaseSKLearnMetric


class PrecisionMetric(BaseSKLearnMetric):

    score_method = recall_score
