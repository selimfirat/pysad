from abc import abstractmethod

import numpy as np

from pysad.core.base_postprocessor import BasePostprocessor

from ._combination import aom, average, maximization, median, moa


class PYODScoreEnsembler(BasePostprocessor):
    """Abstract base class for the scoring ensembling methods for the scoring based ensemblers of the combo library :cite:`zhao2020combo`, which PyOD also uses."""

    @abstractmethod
    def _combine(self, scores):
        """Combines the scores of multiple anomaly detectors.

        Args:
            scores (np.float64 array of shape (1, num_anomaly_detectors)): Scores from multiple anomaly detectors.

        Returns:
            float: Resulting anomaly score.
        """
        pass

    def fit_partial(self, scores):
        """Fits particular (next) timestep's score to train the ensembler. For PYOD based ensemblers, this method does not affect anything and returns self directly.

        Args:
            scores (np.float64 array of shape (num_anomaly_detectors,)): Scores from multiple anomaly detectors.

        Returns:
            object: The fitted ensembler.
        """
        return self

    def transform_partial(self, scores):
        """Combines anomaly scores from multiple anomaly detectors for a particular timestep.

        Args:
            scores (np.float64 array of shape (num_anomaly_detectors,)): Scores from multiple anomaly detectors.

        Returns:
            float: Resulting anomaly score.
        """
        scores = np.asarray(scores, dtype=np.float64).reshape(1, -1)

        return float(self._combine(scores)[0])


class AverageScoreEnsembler(PYODScoreEnsembler):
    """Combines the scores of multiple anomaly detectors for the current instance by taking their weighted average :cite:`zhao2020combo`.

    Args:
        estimator_weights (np.float64 array of shape (1, num_anomaly_detectors) or None): The weights for detectors. If None, the detectors get uniform weights (Default=None).
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
    """Combines the scores of multiple anomaly detectors for the current instance by taking their maximum :cite:`zhao2020combo`."""

    def _combine(self, scores):
        """Wrapping for PyOD the ensembler.

        Args:
            scores (np.float64 array of shape (num_anomaly_detectors, )): List of scores from multiple anomaly detectors.

        Returns:
            float: Resulting anomaly score.
        """
        return maximization(scores)


class MedianScoreEnsembler(PYODScoreEnsembler):
    """Combines the scores of multiple anomaly detectors for the current instance by taking their median :cite:`zhao2020combo`."""

    def _combine(self, scores):
        """Helper method to wrap the PyOD ensembler.

        Args:
            scores (np.float64 array of shape (num_anomaly_detectors, )): List of scores from multiple anomaly detectors.

        Returns:
            float: Resulting anomaly score.
        """
        return median(scores)


class _BucketScoreEnsembler(PYODScoreEnsembler):
    """Base class for the PyOD ensemblers that combine scores in buckets.

    Args:
        n_buckets (int): The number of buckets (Default=5).
        method (str): Either ``'static'`` or ``'dynamic'``. If ``'dynamic'``, the ensembler builds buckets randomly with a dynamic bucket size (Default='static').
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
    """Combines the scores of multiple anomaly detectors by the average of the bucket maxima (AOM) :cite:`aggarwal2015theoretical,zhao2020combo`.

    The ensembler first divides the scores into buckets and takes the maximum for each bucket. Then, the ensembler outputs the average of all these maximum scores of buckets.

    Args:
        n_buckets (int): The number of buckets (Default=5).
        method (str): Either ``'static'`` or ``'dynamic'``. If ``'dynamic'``, the ensembler builds buckets randomly with a dynamic bucket size (Default='static').
        bootstrap_estimators (bool): Whether estimators are drawn with replacement (Default=False).
    """

    _combine_buckets = staticmethod(aom)


class MaximumOfAverageScoreEnsembler(_BucketScoreEnsembler):
    """Combines the scores of multiple anomaly detectors by the maximum of the bucket averages (MOA) :cite:`aggarwal2015theoretical,zhao2020combo`.

    The ensembler first divides the scores into buckets and takes the average for each bucket. Then, the ensembler outputs the maximum of all these average scores of buckets.

    Args:
        n_buckets (int): The number of buckets (Default=5).
        method (str): Either ``'static'`` or ``'dynamic'``. If ``'dynamic'``, the ensembler builds buckets randomly with a dynamic bucket size (Default='static').
        bootstrap_estimators (bool): Whether estimators are drawn with replacement (Default=False).
    """

    _combine_buckets = staticmethod(moa)
