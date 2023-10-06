from pysad.core.base_model import BaseModel
import numpy as np


class KNNCAD(BaseModel):
    """Conformalized density- and distance-based anomaly detection in time-series data :cite:`burnaev2016conformalized`, which uses a combination of a feature extraction method, an approach to assess a score whether a new observation differs significantly from a previously observed data, and a probabilistic interpretation of this score based on the conformal paradigm. This method's implementation is based on `NAB-kNNCAD <https://github.com/numenta/NAB/blob/master/nab/detectors/knncad/knncad_detector.py>`_. This model is univariate.

        Args:
            probationary_period (int): Number of instances in probationary period. Until probationary_period instances are received, the model outputs anomaly score of `0.0`.
    """

    def __init__(self, probationary_period):
        self.buf = []
        self.training = []
        self.calibration = []
        self.scores = []
        self.record_count = 0
        self.pred = -1
        self.k = 27
        self.to_init = True

        self.probationaryPeriod = probationary_period

    def _metric(self, a, b):
        diff = a - np.array(b)

        return np.dot(np.dot(diff, self.sigma), diff.T)

    def _ncm(self, item, item_in_array=False):
        arr = [self._metric(x, item) for x in self.training]

        return np.sum(np.partition(arr, self.k + item_in_array)[:self.k + item_in_array])

    def fit_partial(self, X, y=None):
        """Fits the model to next instance. Note that this model is univariate.

        Args:
            X (np.float64 array of shape (1,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: Returns the self.
        """
        if self.to_init:
            self.dim = 19  # X.shape[0]
            self.sigma = np.diag(np.ones(self.dim))
            self.to_init = False

        self.buf.append(X[0])
        self.record_count += 1
        if len(self.buf) < self.dim:
            return self

        new_item = self.buf[-self.dim:]

        if self.record_count < self.probationaryPeriod:
            self.training.append(new_item)
        else:
            ost = self.record_count % self.probationaryPeriod
            if ost == 0 or ost == int(self.probationaryPeriod / 2):
                try:
                    self.sigma = np.linalg.inv(
                        np.dot(np.array(self.training).T, self.training))
                except np.linalg.linalg.LinAlgError:
                    print('Singular Matrix at record', self.record_count)
            if len(self.scores) == 0:
                self.scores = [self._ncm(v, True) for v in self.training]

            new_score = self._ncm(new_item)

            if self.record_count >= 2 * self.probationaryPeriod:
                self.training.pop(0)
                self.training.append(self.calibration.pop(0))

            self.scores.pop(0)
            self.calibration.append(new_item)
            self.scores.append(new_score)

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (1,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if len(self.buf) < self.dim or self.record_count < self.probationaryPeriod:
            return 0.0

        new_item = self.buf[-self.dim:]

        new_score = self._ncm(new_item)
        result = 1. * len(np.where(np.array(self.scores) < new_score)[0]) / len(self.scores)

        if self.pred > 0:
            self.pred -= 1
            return 0.5
        elif result >= 0.9965:
            self.pred = int(self.probationaryPeriod / 5)

        return result
