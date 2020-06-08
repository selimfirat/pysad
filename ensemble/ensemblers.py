from abc import abstractmethod

from ensemble.base_ensembler import BaseEnsembler
from pyod.models.combination import average, maximization, median, moa, aom

class PYODEnsembler(BaseEnsembler):

    @abstractmethod
    def combine(self, scores):
        pass

    def transform_partial(self, scores):
        scores = scores.reshape(1, -1)

        return self.combine(scores)


class AverageEnsembler(PYODEnsembler):

    def __init__(self, estimator_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.estimator_weights = estimator_weights

    def combine(self, scores):

        return average(scores, estimator_weights=self.estimator_weights)


class MaximumEnsembler(PYODEnsembler):

    def combine(self, scores):

        return maximization(scores)


class MedianEnsembler(PYODEnsembler):

    def combine(self, scores):

        return median(scores)

class AverageOfMaximumEnsembler(PYODEnsembler):

    def __init__(self, n_buckets=5, method='static', bootstrap_estimators=False, **kwargs)
        super().__init__(**kwargs)
        self.method = method
        self.n_buckets = n_buckets
        self.bootstrap_estimators = bootstrap_estimators

    def combine(self, scores):

        return aom(scores, n_buckets=self.n_buckets, method=self.method, bootstrap_estimators=self.bootstrap_estimators)


class MaximumOfAverageEnsembler(PYODEnsembler):


    def __init__(self, n_buckets=5, method='static', bootstrap_estimators=False, **kwargs)
        super().__init__(**kwargs)
        self.method = method
        self.n_buckets = n_buckets
        self.bootstrap_estimators = bootstrap_estimators

    def combine(self, scores):

        return moa(scores, n_buckets=self.n_buckets, method=self.method, bootstrap_estimators=self.bootstrap_estimators)

