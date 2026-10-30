from __future__ import annotations

import math
import numbers
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pysad.core.base_model import BaseModel
from pysad.transform.preprocessing import ShingleTransformer
from pysad.utils.validation import check_univariate


class KNNCAD(BaseModel):
    """Scores each value of a univariate stream with a conformalized k-nearest-neighbor distance over sliding windows :cite:`burnaev2016conformalized`.

    The method combines a feature extraction method, a score that tells whether a new observation differs significantly from the previously observed data, and a probabilistic interpretation of this score based on the conformal paradigm. This method's implementation is based on `NAB-kNNCAD <https://github.com/numenta/NAB/blob/master/nab/detectors/knncad/knncad_detector.py>`_. This model is univariate: each instance must hold exactly one value, and an instance with more raises `ValueError`. Where NAB and the paper disagree, this implementation follows NAB, including: the training and calibration sets and their rotation, the sum of the k smallest quadratic forms with the `inv(XᵀX)` matrix of the uncentered training windows (refreshed every half probationary period) in place of the paper's Mahalanobis distance, and the alarm suppression that returns `0.5`. The paper gives no values for `k` and the window length, so this implementation uses NAB's `k = 27` and `19`. Unlike NAB, it keeps only the last `19` values of the stream, so its memory does not grow with the stream. Each window of `19` values is a shingle of :class:`~pysad.transform.preprocessing.ShingleTransformer`.

    Args:
        probationary_period (int): Number of instances in probationary period. Until probationary_period instances are received, the model outputs anomaly score of `0.0`. Must be a whole number of at least `48` (window length `19` plus `k = 27` plus `2`): the training set holds `probationary_period - 19` windows, and the calibration scores need at least `k + 2` of them. It may be an int (a NumPy integer is accepted, but not a bool) or an integral-valued float, such as the `750.0` that NAB's probation period helper returns, which is converted to an int. Raises `TypeError` for a bool or a non-real type, and `ValueError` for NaN, infinity, a value with a fractional part, or a value below the minimum.
    """

    sigma: np.ndarray

    def __init__(self, probationary_period: float) -> None:
        dim = 19
        k = 27
        min_probationary_period = dim + k + 2

        if isinstance(probationary_period, bool) or not isinstance(
            probationary_period, numbers.Real
        ):
            raise TypeError(
                f"probationary_period must be an int or an integral float, got {probationary_period!r}."
            )

        if not isinstance(probationary_period, numbers.Integral) and not (
            math.isfinite(probationary_period) and float(probationary_period).is_integer()
        ):
            raise ValueError(
                f"probationary_period must be a whole number, got {probationary_period!r}."
            )

        if probationary_period < min_probationary_period:
            raise ValueError(
                f"probationary_period must be at least {min_probationary_period} "
                f"(window length {dim} plus k={k} plus 2), got {probationary_period!r}: "
                f"the training set holds probationary_period - {dim} windows, and the "
                f"calibration scores need at least k + 2 of them."
            )

        probationary_period = int(probationary_period)

        # Each window is a shingle of the last dim values. The model uses no copies at the start.
        self.shingler: ShingleTransformer = ShingleTransformer(dim)
        self.training: list[np.ndarray] = []
        self.calibration: list[np.ndarray] = []
        self.scores: list[float] = []
        self.record_count: int = 0
        self.pred: int = -1
        self.k: int = k
        self.dim: int = dim
        self.to_init: bool = True
        self.probationaryPeriod: int = probationary_period

    @staticmethod
    def _value(X: Any) -> float:
        return check_univariate(X, "KNNCAD")

    def _ncm(self, item: ArrayLike, sigma: np.ndarray, item_in_array: bool = False) -> float:
        # The distances (x - item) sigma (x - item)^T of all the training windows x at once.
        # einsum does not call BLAS, which can raise false floating-point flags on macOS.
        diff = np.asarray(self.training, dtype=np.float64) - np.asarray(item, dtype=np.float64)
        arr = np.einsum("ij,ij->i", np.einsum("ij,jk->ik", diff, sigma), diff)

        return np.sum(np.partition(arr, self.k + item_in_array)[: self.k + item_in_array])

    def _sigma_at(self, record_count: int) -> np.ndarray:
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

    def _calibration_scores(self, sigma: np.ndarray) -> list[float]:
        """Returns the calibration scores, computed from the training set with the given sigma when there are none yet.

        Args:
            sigma (np.float64 array of shape (dim, dim)): The sigma for the current record.

        Returns:
            list: The calibration scores.
        """
        if len(self.scores) == 0:
            return [self._ncm(v, sigma, True) for v in self.training]

        return self.scores

    def _fit(self, value: float, new_score: float | None = None) -> None:
        # The fit of fit_partial. ``new_score`` is the nonconformity score of the window that ends
        # with ``value`` if _score computed it already for the same record, else None.
        if self.to_init:
            self.sigma = np.diag(np.ones(self.dim))
            self.to_init = False

        full = self.shingler.full
        new_item = self.shingler.fit_transform_partial(np.array([value]))
        self.record_count += 1
        if not full:
            return

        if self.record_count < self.probationaryPeriod:
            self.training.append(new_item)
        else:
            try:
                self.sigma = self._sigma_at(self.record_count)
            except np.linalg.LinAlgError:
                print("Singular Matrix at record", self.record_count)
            self.scores = self._calibration_scores(self.sigma)

            if new_score is None:
                new_score = self._ncm(new_item, self.sigma)

            if self.record_count >= 2 * self.probationaryPeriod:
                self.training.pop(0)
                self.training.append(self.calibration.pop(0))

            self.scores.pop(0)
            self.calibration.append(new_item)
            self.scores.append(new_score)

    def _score(self, value: float) -> tuple[float, float | None]:
        # The score of score_partial, and the nonconformity score of the window that ends with
        # ``value`` (None while the model only learns), which _fit can reuse for the same record.
        if self.to_init or not self.shingler.full:
            return 0.0, None

        record_count = self.record_count + 1
        if record_count < self.probationaryPeriod:
            return 0.0, None

        new_item = self.shingler.transform_partial(np.array([value]))

        try:
            sigma = self._sigma_at(record_count)
        except np.linalg.LinAlgError:
            sigma = self.sigma
        scores = self._calibration_scores(sigma)

        new_score = self._ncm(new_item, sigma)

        return 1.0 * len(np.where(np.array(scores) < new_score)[0]) / len(scores), new_score

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> KNNCAD:
        """Updates the model with the next instance. Note that this model is univariate.

        Args:
            X (float or np.float64 array): The instance to fit, holding exactly one value (e.g. a scalar or an array of shape (1,) or (1, 1)).
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.

        Raises:
            ValueError: If `X` does not hold exactly one value.
        """
        value = self._value(X)
        self._fit(value)

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the window that ends with the given instance, i.e., the last `dim - 1` fitted values followed by `X`, against the current calibration scores. The score is the fraction of calibration scores lower than the window's score. This method does not change the model. Alarm suppression, which outputs `0.5` for a while after an alarm as in NAB, applies only in `fit_score_partial`.

        Args:
            X (float or np.float64 array): The instance to score, holding exactly one value (e.g. a scalar or an array of shape (1,) or (1, 1)). Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.

        Raises:
            ValueError: If `X` does not hold exactly one value.
        """
        value = self._value(X)
        return self._score(value)[0]

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Scores the window that ends with the given instance and then fits the model to it, as NAB's detector does for each record. After a score of at least `0.9965` raises an alarm, the next `int(probationary_period / 5)` scores are suppressed to `0.5`.

        Args:
            X (float or np.float64 array): The instance to fit and score, holding exactly one value (e.g. a scalar or an array of shape (1,) or (1, 1)).
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance.

        Raises:
            ValueError: If `X` does not hold exactly one value; the model is left unchanged.
        """
        value = self._value(X)
        score, new_score = self._score(value)
        self._fit(value, new_score)

        if self.pred > 0:
            self.pred -= 1
            return 0.5
        elif score >= 0.9965:
            self.pred = int(self.probationaryPeriod / 5)

        return score
