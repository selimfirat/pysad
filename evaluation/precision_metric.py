from sklearn.metrics import precision_score
from evaluation.base_sklearn_metric import BaseSKLearnMetric


class PrecisionMetric(BaseSKLearnMetric):

    score_method = precision_score
