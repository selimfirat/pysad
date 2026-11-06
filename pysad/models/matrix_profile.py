from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _int_at_least, check_univariate


class MatrixProfile(BaseModel):
    """Scores each value of a univariate stream with the matrix profile value of the subsequence that ends at it, as STAMPI computes it :cite:`yeh2016matrix`.

    The subsequence of the last ``window_length`` values, which ends with the scored value, gets the z-normalized Euclidean distance to its nearest neighbor among the earlier subsequences. This is the new matrix profile entry that STAMPI appends for each arriving value. A high distance means that the shape is unlike any earlier shape, a time series discord, even if each value is normal. As in the paper, the distance ignores the trivial matches, the subsequences that start at most ``ceil(window_length / 2)`` values before it. This model is univariate: each instance must hold exactly one value, and an instance with more raises `ValueError`.

    Deviations from the paper:

    - STAMPI keeps the whole stream. This model keeps only the last ``history_size`` values, so it compares the new subsequence only with the earlier subsequences that lie within those values and the scored value. This is the ``egress`` mode of the STUMPY ``stumpi`` class.
    - STAMPI also lowers the profile entries of earlier subsequences when the new subsequence is closer to them. The score of each value is fixed when it arrives, so this model does not keep the earlier entries.
    - The paper computes the distance profile with MASS, an FFT. This model computes the same dot products directly, which costs ``O(history_size * window_length)`` for each value.
    - The paper does not define the distance for a constant subsequence. As in STUMPY, the distance between two constant subsequences is 0, and the distance between a constant subsequence and another subsequence is ``sqrt(window_length)``.

    The score is 0 until the model has at least one earlier subsequence outside the trivial-match zone, that is, for the first ``window_length + ceil(window_length / 2)`` values.

    Args:
        window_length (int): The subsequence length ``m``. Must be at least 2.
        history_size (int): The number of most recent values that the model keeps. Must be at least ``window_length + ceil(window_length / 2)`` (Default=1000).
    """

    def __init__(self, window_length: int, history_size: int = 1000) -> None:
        self.window_length = _int_at_least(window_length, "window_length", 2)
        self.exclusion_zone = math.ceil(self.window_length / 2)
        self.history_size = _int_at_least(
            value=history_size,
            name="history_size",
            minimum=self.window_length + self.exclusion_zone,
        )

        self.values_ = np.zeros(self.history_size)
        self.num_values_ = 0

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> MatrixProfile:
        """Appends the next value to the history and drops the oldest value once the history is full.

        Args:
            X (float or np.float64 array): The instance to fit, holding exactly one value (e.g. a scalar or an array of shape (1,)).
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.

        Raises:
            ValueError: If `X` does not hold exactly one value.
        """
        value = check_univariate(X, "MatrixProfile")
        if self.num_values_ < self.history_size:
            self.values_[self.num_values_] = value
            self.num_values_ += 1
        else:
            self.values_[:-1] = self.values_[1:]
            self.values_[-1] = value

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the subsequence of the last ``window_length - 1`` fitted values followed by `X`. This method does not change the model.

        Args:
            X (float or np.float64 array): The instance to score, holding exactly one value (e.g. a scalar or an array of shape (1,)). Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The z-normalized Euclidean distance from the subsequence to its nearest earlier neighbor.

        Raises:
            ValueError: If `X` does not hold exactly one value.
        """
        value = check_univariate(X, "MatrixProfile")
        m = self.window_length
        # The earlier subsequences end before the trivial-match zone of the new one.
        num_neighbors = self.num_values_ + 1 - m - self.exclusion_zone
        if num_neighbors <= 0:
            return 0.0

        values = np.append(self.values_[: self.num_values_], value)
        query = values[-m:]
        neighbors = sliding_window_view(values[: num_neighbors + m - 1], m)  # (num_neighbors, m)

        means, stds = neighbors.mean(axis=1), neighbors.std(axis=1)
        constant = neighbors.max(axis=1) == neighbors.min(axis=1)
        query_constant = query.max() == query.min()
        with np.errstate(divide="ignore", invalid="ignore"):
            correlations = (neighbors @ query - m * means * query.mean()) / (m * stds * query.std())
        squared = 2 * m * (1 - np.minimum(correlations, 1.0))
        squared = np.where(constant | query_constant, m, squared)
        squared[constant & query_constant] = 0.0

        return math.sqrt(squared.min())
