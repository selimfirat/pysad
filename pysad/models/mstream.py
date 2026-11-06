from __future__ import annotations

import math

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _int_at_least


class MStream(BaseModel):
    """Scores each record by the chi-squared statistic of its decayed bucket counts in the current time tick against all earlier ticks (MStream) :cite:`bhatia2021mstream`.

    This is MStream (Algorithm 3 of the paper) for records with real-valued features. FeatureHash (Algorithm 1) maps each feature on its own into one of ``num_buckets`` buckets: it takes ``log(1 + x)``, scales the result to [0, 1] by the running minimum and maximum of the feature, and keeps the bucket ``floor(value * num_buckets) mod num_buckets``. RecordHash (Algorithm 2) maps the whole scaled record into a bucket of each of ``num_hash_fns`` count-min sketch rows. Each row has ``ceil(log2(num_buckets))`` random Gaussian vectors, and the signs of the dot products with them form the bits of the bucket index. For each bucket, the model keeps the total count ``s`` of all records so far and the current count ``a``, which it multiplies by ``alpha`` at the start of every time tick instead of resetting it. The record count of a sketch is its smallest count over the rows. The score of a record in tick ``t`` is the sum of ``(a - s / t)**2 * t**2 / (s * (t - 1))`` (Equation 1) over the record bucket and the ``num_features`` feature buckets (Definition 1), from the counts that already hold the record. The records of the first tick score 0.0, since the statistic needs a past tick.

    The paper takes a timestamp with each record, and the records with the same timestamp form a tick. A PySAD model gets no timestamp, so every ``tick_size`` records form a tick. The paper does the same for KDDCUP99, which has no timestamps, with 1000 records per tick. The paper inserts a record and then reads its counts. ``score_partial`` computes the same score without the insertion, from the counts and the feature ranges that the record would give. So ``fit_score_partial``, which scores a record before it fits it, gives the score of the paper.

    The implementation differs from the reference implementation (https://github.com/Stream-AD/MStream, Apache-2.0) as follows, because the paper differs:

    - The reference maps a scaled value to the bucket ``floor(value * (num_buckets - 1))``. The model uses the bucket of the paper, so the largest value so far shares bucket 0 with the smallest one.
    - The reference squares ``max(0, a - s / t)``, so that only a count above its mean adds to the score. The model squares ``a - s / t``, as Equation 1 does.
    - The reference sets a bit for a dot product of 0. The model sets it only for a positive dot product, as the paper does.
    - The reference outputs ``log(1 + score)``, which ranks the records in the same order. The model outputs the score of Definition 1.
    - The reference draws the random vectors of the current and the total record counts from two generators with separate random seeds, so the two counts of a record come from different buckets. The model uses one RecordHash for both counts, as the paper describes.

    The paper leaves open which values RecordHash takes. The model uses the scaled values, as the reference does. It also scales a feature with equal minimum and maximum to 0, as the reference does. The paper handles categorical features too, but the model takes real-valued features only.

    Memory is fixed: two counts for each of ``num_buckets`` buckets of every feature and of every sketch row, and the random vectors.

    Args:
        num_hash_fns (int): The number of count-min sketch rows of RecordHash, each with its own random vectors (Default=2, as in the paper).
        num_buckets (int): The number of buckets ``b`` of each hash. With continuous features, almost every value gets its own bucket of 1024. Fewer buckets let similar values share a bucket (Default=1024, as in the paper).
        alpha (float): The temporal decay factor in (0, 1). The current counts are multiplied by it at the start of each tick (Default=0.85, the paper's value for KDDCUP99).
        tick_size (int): The number of records in each time tick (Default=1).
        log_transform (bool): Whether FeatureHash takes ``log(1 + x)`` of each feature before it scales the feature, as in the paper. The transform needs every value above -1. Set it to False for data with other values (Default=True).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the random vectors of RecordHash. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).

    Raises:
        ValueError: If ``alpha`` is not in (0, 1).
    """

    def __init__(
        self,
        num_hash_fns: int = 2,
        num_buckets: int = 1024,
        alpha: float = 0.85,
        tick_size: int = 1,
        log_transform: bool = True,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        self.num_hash_fns = _int_at_least(num_hash_fns, "num_hash_fns", 1)
        self.num_buckets = _int_at_least(num_buckets, "num_buckets", 1)
        self.alpha = alpha
        self.tick_size = _int_at_least(tick_size, "tick_size", 1)
        self.log_transform = log_transform
        self.random_state = random_state
        self.num_fitted_ = 0

    def _transform(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64)
        if not self.log_transform:
            return x
        if (x <= -1.0).any():
            raise ValueError(
                "MStream takes log(1 + x) of each feature, so every value must be above -1. "
                "Pass log_transform=False for other data."
            )
        return np.log1p(x)

    def _buckets(
        self, x: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The feature buckets, shape (num_features,), and the record buckets, shape (num_hash_fns,), of a transformed record."""
        span = maximum - minimum
        scaled = np.divide(x - minimum, span, out=np.zeros_like(x), where=span > 0.0)
        feature_buckets = (scaled * self.num_buckets).astype(np.intp) % self.num_buckets
        bits = self.hyperplanes_ @ scaled > 0.0  # (num_hash_fns, num_bits)
        record_buckets = (bits @ self._bit_values) % self.num_buckets
        return feature_buckets, record_buckets

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> MStream:
        """Updates the bucket counts with the next record.

        Args:
            X (np.float64 array of shape (num_features,)): The record to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        x = self._transform(X)
        if self.num_fitted_ == 0:
            num_features = x.shape[0]
            num_bits = math.ceil(math.log2(self.num_buckets))
            self.hyperplanes_ = check_random_state(self.random_state).normal(
                size=(self.num_hash_fns, num_bits, num_features)
            )
            self._bit_values = 2 ** np.arange(num_bits)
            self.min_ = x.copy()
            self.max_ = x.copy()
            self.feature_current_ = np.zeros((num_features, self.num_buckets))
            self.feature_total_ = np.zeros((num_features, self.num_buckets))
            self.record_current_ = np.zeros((self.num_hash_fns, self.num_buckets))
            self.record_total_ = np.zeros((self.num_hash_fns, self.num_buckets))
            self._features = np.arange(num_features)
            self._rows = np.arange(self.num_hash_fns)
        elif self.num_fitted_ % self.tick_size == 0:
            # A new tick starts: the current counts decay instead of resetting.
            self.feature_current_ *= self.alpha
            self.record_current_ *= self.alpha
        np.minimum(self.min_, x, out=self.min_)
        np.maximum(self.max_, x, out=self.max_)

        feature_buckets, record_buckets = self._buckets(x, self.min_, self.max_)
        self.feature_current_[self._features, feature_buckets] += 1.0
        self.feature_total_[self._features, feature_buckets] += 1.0
        self.record_current_[self._rows, record_buckets] += 1.0
        self.record_total_[self._rows, record_buckets] += 1.0
        self.num_fitted_ += 1

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the next record by the counts it would have after the model fits it. This method does not change the model.

        Args:
            X (np.float64 array of shape (num_features,)): The record to score.

        Returns:
            float: The anomalousness score of the record, at least 0.0. Higher scores represent more anomalous records.
        """
        x = self._transform(X)
        t = self.num_fitted_ // self.tick_size + 1
        if t == 1:
            return 0.0
        decay = self.alpha if self.num_fitted_ % self.tick_size == 0 else 1.0
        feature_buckets, record_buckets = self._buckets(
            x, np.minimum(self.min_, x), np.maximum(self.max_, x)
        )
        # The counts of the record bucket and of the feature buckets, after the record is added.
        current = np.append(
            self.feature_current_[self._features, feature_buckets],
            self.record_current_[self._rows, record_buckets].min(),
        )
        current = decay * current + 1.0
        total = (
            np.append(
                self.feature_total_[self._features, feature_buckets],
                self.record_total_[self._rows, record_buckets].min(),
            )
            + 1.0
        )
        return float(((current - total / t) ** 2 / total).sum() * t * t / (t - 1))
