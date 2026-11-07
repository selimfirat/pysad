from __future__ import annotations

import numpy as np

from pysad.core.base_model import BaseModel
from pysad.models.matrix_profile import _nearest_distance
from pysad.utils.validation import _int_at_least, check_univariate


class DAMP(BaseModel):
    """Scores each value of a univariate stream with the approximate left matrix profile value of the subsequence that ends at it, as the online X-Lag-Amnesic DAMP computes it :cite:`lu2022matrix`.

    The left matrix profile value of a subsequence is the z-normalized Euclidean distance to its nearest neighbor among the earlier subsequences. As in Definition 8 of the paper, the earlier subsequences end at or before the first value of the newest subsequence. DAMP keeps the best-so-far discord score, the highest exact value that it found so far. The backward search looks for a neighbor in the last ``2^ceil(log2(window_length))`` values and doubles this length until it finds a neighbor closer than the best-so-far score. If the search reaches the oldest value that it may read, the score is the exact left matrix profile value, and a higher score becomes the new best-so-far score. Otherwise, the score lies between the exact value and the best-so-far score. Thus each new left-discord gets its exact score, and most values cost only a short search. But as the paper notes, a score below the best-so-far score is only an upper bound, so a high score that does not set a new best-so-far score does not always mark an anomaly. This model is univariate: each instance must hold exactly one value, and an instance with more raises `ValueError`.

    The paper defines the online algorithm as DAMP with a lookahead of 0, which has no forward pruning. This model implements that algorithm. It also bounds the backward search to the last ``history_size`` values before the newest subsequence, as the X-Lag-Amnesic DAMP variant of the paper does with its lag X. So the model keeps only the last ``history_size + window_length - 1`` values.

    Deviations from the paper:

    - The paper computes the distances with MASS, an FFT. This model computes the same dot products directly, as :class:`~pysad.models.MatrixProfile` does.
    - The paper does not define the distance for a constant subsequence, and the reference code rejects such a series. As in :class:`~pysad.models.MatrixProfile` and STUMPY, the distance between two constant subsequences is 0, and the distance between a constant subsequence and another subsequence is ``sqrt(window_length)``.

    The authors' MATLAB code differs from the paper in three ways. It starts the backward search with ``2^ceil(log2(8 * window_length))`` values and then searches blocks of older values, it computes the exact score of the first ``16 * window_length`` subsequences, and it rejects a series with constant stretches. The best-so-far score is the highest exact score in both, so these differences change only the scores below the best-so-far score. On a noisy sine with an inserted discord, the scores that set a new best-so-far score and the top discord match an unofficial Python port of that code to ``1e-14``.

    As the split point between the training and the test data of the paper, the first ``probationary_period`` values only serve as neighbors. They get the score 0 and do not raise the best-so-far score. Without them, the first subsequences have only a few neighbors, so their high scores would set a best-so-far score that later discords do not reach.

    Args:
        window_length (int): The subsequence length ``m``. Must be at least 2.
        history_size (int): The number of values before the newest subsequence that the backward search can read, the lag X of X-Lag-Amnesic DAMP. Must be at least ``window_length - 1`` (Default=10000).
        probationary_period (int): The number of first values that get the score 0, the split point of the paper. Must be at least ``2 * window_length - 2``, the number of values before the first subsequence with an earlier neighbor (Default=None, which uses ``4 * window_length``, the shortest training data that the reference code recommends).
    """

    def __init__(
        self,
        window_length: int,
        history_size: int = 10000,
        probationary_period: int | None = None,
    ) -> None:
        self.window_length = _int_at_least(window_length, "window_length", 2)
        self.history_size = _int_at_least(history_size, "history_size", self.window_length - 1)
        self.probationary_period = _int_at_least(
            value=4 * self.window_length if probationary_period is None else probationary_period,
            name="probationary_period",
            minimum=2 * self.window_length - 2,
        )
        # The smallest power of 2 at least window_length, the first length of the backward search.
        self.initial_prefix = 1 << (self.window_length - 1).bit_length()

        self.values_ = np.zeros(self.history_size + self.window_length - 1)
        self.num_values_ = 0
        self.best_so_far_ = 0.0

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> DAMP:
        """Updates the best-so-far discord score with the score of `X`, and appends `X` to the history.

        Args:
            X (float or np.float64 array): The instance to fit, holding exactly one value (e.g. a scalar or an array of shape (1,)).
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.

        Raises:
            ValueError: If `X` does not hold exactly one value.
        """
        value = check_univariate(X, "DAMP")
        self._learn(value, self._backward_processing(value))
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the subsequence of the last ``window_length - 1`` fitted values followed by `X`. This method does not change the model.

        Args:
            X (float or np.float64 array): The instance to score, holding exactly one value (e.g. a scalar or an array of shape (1,)). Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The approximate left matrix profile value of the subsequence.

        Raises:
            ValueError: If `X` does not hold exactly one value.
        """
        return self._backward_processing(check_univariate(X, "DAMP"))

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Scores `X` and then fits it, with one backward search.

        Args:
            X (float or np.float64 array): The instance to fit and score, holding exactly one value (e.g. a scalar or an array of shape (1,)).
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The approximate left matrix profile value of the subsequence that ends at `X`.

        Raises:
            ValueError: If `X` does not hold exactly one value.
        """
        value = check_univariate(X, "DAMP")
        score = self._backward_processing(value)
        self._learn(value, score)
        return score

    def _backward_processing(self, value: float) -> float:
        """Returns the approximate left matrix profile value of the subsequence that ends at `value` (Table 2 and Table 4 of the paper)."""
        if self.num_values_ < self.probationary_period:
            return 0.0

        n = min(self.num_values_, self.values_.size)
        # The newest subsequence starts at the fitted index n + 1 - m. Its neighbors end at or before it.
        end = n + 2 - self.window_length

        query = np.append(self.values_[end - 1 : n], value)
        prefix = self.initial_prefix
        while end > prefix:
            distance = _nearest_distance(series=self.values_[end - prefix : end], query=query)
            if distance < self.best_so_far_:
                return distance

            prefix *= 2

        # The search reached the oldest value that it may read, so the distance is exact.
        return _nearest_distance(series=self.values_[:end], query=query)

    def _learn(self, value: float, score: float) -> None:
        """Raises the best-so-far score to `score` and appends `value`, dropping the oldest value once the history is full."""
        self.best_so_far_ = max(self.best_so_far_, score)
        if self.num_values_ < self.values_.size:
            self.values_[self.num_values_] = value
        else:
            self.values_[:-1] = self.values_[1:]
            self.values_[-1] = value
        self.num_values_ += 1
