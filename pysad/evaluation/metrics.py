from sklearn.metrics import recall_score, precision_score, roc_auc_score, average_precision_score
from pysad.evaluation.base_sklearn_metric import BaseSKLearnMetric


class PrecisionMetric(BaseSKLearnMetric):
    """Precision wrapper class for sklearn.
    """

    def _evaluate(self, y_true, y_pred):
        return precision_score(y_true, y_pred)


class RecallMetric(BaseSKLearnMetric):
    """Recall wrapper class for sklearn.
    """

    def _evaluate(self, y_true, y_pred):
        return recall_score(y_true, y_pred)


class AUROCMetric(BaseSKLearnMetric):
    """Area under roc curve wrapper class for sklearn.
    """

    def _evaluate(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)


class AUPRMetric(BaseSKLearnMetric):
    """Area under PR curve wrapper class for sklearn.
    """

    def _evaluate(self, y_true, y_pred):
        return average_precision_score(y_true, y_pred)
