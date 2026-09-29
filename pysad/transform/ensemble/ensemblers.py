from abc import abstractmethod

import numpy as np
from pyod.models.combination import aom, average, maximization, median, moa

from pysad.core.base_postprocessor import BasePostprocessor


class PYODScoreEnsembler(BasePostprocessor):
    """Abstract base class for the scoring ensembling methods for the scoring based ensemblers of the `PyOD <https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.combination>`_."""

    @abstractmethod
    def _combine(self, scores):
        """Abstract method that directly uses  of our framework to be filled.

        Args:
            scores: np.float64 array of shape (1, num_scores)
                List of scores from multiple anomaly detectors.

        Returns:
            float: Resulting anomaly score.
        """
        pass

    def fit_partial(self, scores):
        """Fits particular (next) timestep's score to train the ensembler. For PYOD based ensemblers, this method does not affect anything and returns self directly.

        Args:
            scores: np.float64 array of shape (num_anomaly_detectors, )
                List of scores from multiple anomaly detectors.

        Returns:
            object: The fitted ensembler.
        """
        return self

    def transform_partial(self, scores):
        """Combines anomaly scores from multiple anomaly detectors for a particular timestep.

        Args:
            scores: np.float64 array of shape (num_anomaly_detectors, )
                List of scores from multiple anomaly detectors.

        Returns:
            float: Resulting anomaly score.
        """
        scores = np.asarray(scores, dtype=np.float64).reshape(1, -1)

        return float(self._combine(scores)[0])


class AverageScoreEnsembler(PYODScoreEnsembler):
    """An wrapper class that results in the weighted average of the anomaly scores from multiple anomaly detectors. For more details, see `PyOD documentation <https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.combination>`_.

    Args:
        estimator_weights (np array of shape (1, num_anomaly_detectors)): The weights for detectors. If None, uniform weights are assigned.

    """

    def __init__(self, estimator_weights=None):

        self.estimator_weights = estimator_weights

    def _combine(self, scores):
        """Wrapping for PyOD the ensembler.

        Args:
            scores (np.float64 array of shape (num_anomaly_detectors, )): List of scores from multiple anomaly detectors.

        Returns:
            float: Resulting anomaly score.
        """
        return average(scores, estimator_weights=self.estimator_weights)


class MaximumScoreEnsembler(PYODScoreEnsembler):
    """An ensembler that results the maximum of the previous scores."""

    def _combine(self, scores):
        """
        Wrapping for PyOD the ensembler.
        Args:
            scores (np.float64 array of shape (num_anomaly_detectors, )) List of scores from multiple anomaly detectors.

        Returns:
            float: Resulting anomaly score.
        """
        return maximization(scores)


class MedianScoreEnsembler(PYODScoreEnsembler):
    """An ensembler that results the median of the previous scores."""

    def _combine(self, scores):
        """
        Helper method to wrap the PyOD ensembler.
        Args:
            scores (np.float64 array of shape (num_anomaly_detectors, )) : List of scores from multiple anomaly detectors.

        Returns:
            float: Resulting anomaly score.
        """
        return median(scores)


class _BucketScoreEnsembler(PYODScoreEnsembler):
    """Base class for the PyOD ensemblers that combine scores in buckets.

    Args:
        n_buckets (int): The number of subgroups to build (Default=5).
        method (str):  {'static', 'dynamic'}, if 'dynamic', build subgroups randomly with dynamic bucket size (Default='static').
        bootstrap_estimators (bool): Whether estimators are drawn with replacement (Default=False).
    """

    def __init__(self, n_buckets=5, method="static", bootstrap_estimators=False):
        self.method = method
        self.n_buckets = n_buckets
        self.bootstrap_estimators = bootstrap_estimators

    def _combine(self, scores):
        """Wrapping for PyOD the ensembler.

        Args:
            scores (np.float64 array of shape (num_anomaly_detectors, )): List of scores from multiple anomaly detectors.

        Returns:
            float: Resulting anomaly score.
        """
        return self._combine_buckets(
            scores,
            n_buckets=self.n_buckets,
            method=self.method,
            bootstrap_estimators=self.bootstrap_estimators,
        )


class AverageOfMaximumScoreEnsembler(_BucketScoreEnsembler):
    """Average of maximum scores ensembler that outputs the average of maximum. For more details, see :cite:`aggarwal2015theoretical` and `PyOD documentation <https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.combination>`_. The ensembler first divides the scores into buckets and takes the maximum for each bucket. Then, the ensembler outputs the average of all these maximum scores of buckets.

    Args:
        n_buckets (int): The number of subgroups to build (Default=5).
        method (str):  {'static', 'dynamic'}, if 'dynamic', build subgroups randomly with dynamic bucket size (Default='static').
        bootstrap_estimators (bool): Whether estimators are drawn with replacement (Default=False).
    """

    _combine_buckets = staticmethod(aom)


class MaximumOfAverageScoreEnsembler(_BucketScoreEnsembler):
    """Maximum of average scores ensembler that outputs the maximum of average. For more details, see :cite:`aggarwal2015theoretical` and `PyOD documentation <https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.combination>`_. The ensembler first divides the scores into buckets and takes the average for each bucket. Then, the ensembler outputs the maximum of all these average scores of buckets.

    Args:
        n_buckets (int): The number of subgroups to build (Default=5).
        method (str): {'static', 'dynamic'}, if 'dynamic', build subgroups randomly with dynamic bucket size (Default='static').
        bootstrap_estimators (bool): Whether estimators are drawn with replacement (Default=False).
    """

    _combine_buckets = staticmethod(moa)
