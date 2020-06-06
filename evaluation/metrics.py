from sklearn.metrics import recall_score, precision_score, roc_auc_score, average_precision_score
from evaluation.base_sklearn_metric import BaseSKLearnMetric


class PrecisionMetric(BaseSKLearnMetric):

    score_method = precision_score


class RecallMetric(BaseSKLearnMetric):

    score_method = recall_score


class AUROCMetric(BaseSKLearnMetric):

    score_method = roc_auc_score


class AUPRMetric(BaseSKLearnMetric):

    score_method = average_precision_score
