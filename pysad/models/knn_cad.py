import numbers

from pysad.core.base_model import BaseModel
import numpy as np


class KNNCAD(BaseModel):
    """Conformalized density- and distance-based anomaly detection in time-series data :cite:`burnaev2016conformalized`, which uses a combination of a feature extraction method, an approach to assess a score whether a new observation differs significantly from a previously observed data, and a probabilistic interpretation of this score based on the conformal paradigm. This method's implementation is based on `NAB-kNNCAD <https://github.com/numenta/NAB/blob/master/nab/detectors/knncad/knncad_detector.py>`_. This model is univariate. Where NAB and the paper disagree, this implementation follows NAB, including: the training and calibration sets and their rotation, the sum of squared quadratic forms with the `inv(XᵀX)` distance (refreshed every half probationary period) in place of the paper's Eq. (1) distance sum, the fixed `k = 27` and window length `19`, and the alarm suppression that returns `0.5`.

        Args:
            probationary_period (int): Number of instances in probationary period. Until probationary_period instances are received, the model outputs anomaly score of `0.0`. Must be an int (a NumPy integer is accepted, but not a bool) of at least `48` (window length `19` plus `k = 27` plus `2`): the training set holds `probationary_period - 19` windows, and the calibration scores need at least `k + 2` of them. Raises `TypeError` if not an int and `ValueError` if below the minimum.
    """

    def __init__(self, probationary_period):
        dim = 19
        k = 27
        min_probationary_period = dim + k + 2

        if isinstance(probationary_period, bool) or not isinstance(probationary_period, numbers.Integral):
            raise TypeError(
                f"probationary_period must be an int, got {probationary_period!r}."
            )

        probationary_period = int(probationary_period)
        if probationary_period < min_probationary_period:
            raise ValueError(
                f"probationary_period must be at least {min_probationary_period} "
                f"(window length {dim} plus k={k} plus 2): the training "
                f"set holds probationary_period - {dim} windows, and the "
                f"calibration scores need at least k + 2 of them."
            )

        self.buf = []
        self.training = []
        self.calibration = []
        self.scores = []
        self.record_count = 0
        self.pred = -1
        self.k = k
        self.dim = dim
        self.to_init = True
        self.probationaryPeriod = probationary_period

    def _metric(self, a, b, sigma):
        diff = a - np.array(b)

        return np.dot(np.dot(diff, sigma), diff.T)

    def _ncm(self, item, sigma, item_in_array=False):
        arr = [self._metric(x, item, sigma) for x in self.training]

        return np.sum(np.partition(arr, self.k + item_in_array)[:self.k + item_in_array])

    def _sigma_at(self, record_count):
        """Returns the sigma NAB uses at the given record, recomputed from the training set when a refresh is due.

        Args:
            record_count (int): The number of the record, counting from 1.

        Returns:
            np.float64 array of shape (dim, dim): The sigma for the record.

        Raises:
            np.linalg.LinAlgError: If a refresh is due and the training set gives a singular matrix.
        """
        ost = record_count % self.probationaryPeriod
        if ost == 0 or ost == int(self.probationaryPeriod / 2):
            return np.linalg.inv(np.dot(np.array(self.training).T, self.training))

        return self.sigma

    def _calibration_scores(self, sigma):
        """Returns the calibration scores, computed from the training set with the given sigma when there are none yet.

        Args:
            sigma (np.float64 array of shape (dim, dim)): The sigma for the current record.

        Returns:
            list: The calibration scores.
        """
        if len(self.scores) == 0:
            return [self._ncm(v, sigma, True) for v in self.training]

        return self.scores

    def fit_partial(self, X, y=None):
        """Fits the model to next instance. Note that this model is univariate.

        Args:
            X (np.float64 array of shape (1,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: Returns the self.
        """
        if self.to_init:
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
            try:
                self.sigma = self._sigma_at(self.record_count)
            except np.linalg.LinAlgError:
                print('Singular Matrix at record', self.record_count)
            self.scores = self._calibration_scores(self.sigma)

            new_score = self._ncm(new_item, self.sigma)

            if self.record_count >= 2 * self.probationaryPeriod:
                self.training.pop(0)
                self.training.append(self.calibration.pop(0))

            self.scores.pop(0)
            self.calibration.append(new_item)
            self.scores.append(new_score)

        return self

    def score_partial(self, X):
        """Scores the window that ends with the given instance, i.e., the last `dim - 1` fitted values followed by `X`, against the current calibration scores. The score is the fraction of calibration scores lower than the window's score. This method does not change the model. Alarm suppression, which outputs `0.5` for a while after an alarm as in NAB, applies only in `fit_score_partial`.

        Args:
            X (np.float64 array of shape (1,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.to_init or len(self.buf) + 1 < self.dim:
            return 0.0

        record_count = self.record_count + 1
        if record_count < self.probationaryPeriod:
            return 0.0

        new_item = self.buf[-(self.dim - 1):] + [X[0]]

        try:
            sigma = self._sigma_at(record_count)
        except np.linalg.LinAlgError:
            sigma = self.sigma
        scores = self._calibration_scores(sigma)

        new_score = self._ncm(new_item, sigma)

        return 1. * len(np.where(np.array(scores) < new_score)[0]) / len(scores)

    def fit_score_partial(self, X, y=None):
        """Scores the window that ends with the given instance and then fits the model to it, as NAB's detector does for each record. After a score of at least `0.9965` raises an alarm, the next `probationary_period / 5` scores are suppressed to `0.5`.

        Args:
            X (np.float64 array of shape (1,)): The instance to fit and score.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance.
        """
        score = self.score_partial(X)
        self.fit_partial(X, y)

        if self.pred > 0:
            self.pred -= 1
            return 0.5
        elif score >= 0.9965:
            self.pred = int(self.probationaryPeriod / 5)

        return score
