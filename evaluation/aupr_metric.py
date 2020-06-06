from sklearn.metrics import average_precision_score
from evaluation.base_sklearn_metric import BaseSKLearnMetric


class AUPRMetric(BaseSKLearnMetric):

    score_method = average_precision_score
