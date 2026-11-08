from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _int_at_least

# The prime that the reference implementation uses to combine the source and the destination into one key.
_EDGE_KEY_PRIME = 104729
# The Mersenne prime 2**61 - 1 of the pairwise-independent hash functions.
_HASH_PRIME = 2**61 - 1


class MIDAS(BaseModel):
    """Scores each edge of a graph stream by the chi-squared statistic of its count in the current time tick against all earlier ticks (MIDAS) :cite:`bhatia2020midas`.

    This is MIDAS (Algorithm 1 of the paper). Each instance is an edge ``[source, destination]`` of integer node IDs. Two count-min sketches with the same ``num_hash_fns`` rows of ``num_buckets`` buckets keep, for each edge ``(u, v)``, the total count ``s`` of the edge so far and its count ``a`` in the current time tick. The current counts reset to 0 at the start of every tick. The count of an edge in a sketch is its smallest count over the rows. The score of an edge in tick ``t`` is ``(a - s / t)**2 * t**2 / (s * (t - 1))`` (Definition 1), from the counts that already hold the edge. The edges of the first tick score 0.0, since the statistic needs a past tick.

    The paper takes a timestamp with each edge, and the edges with the same timestamp form a tick. A PySAD model gets no timestamp, so every ``tick_size`` edges form a tick, as in :class:`~pysad.models.MStream`. The paper inserts an edge and then reads its counts. ``score_partial`` computes the same score without the insertion, from the counts that the edge would give. So ``fit_score_partial``, which scores an edge before it fits it, gives the score of the paper.

    The paper does not give the hash functions. As in the reference implementation (https://github.com/Stream-AD/MIDAS, Apache-2.0), the key of an edge is ``u + 104729 * v``. Row ``i`` maps the key to the bucket ``((p_i * key + q_i) mod P) mod num_buckets``, with the prime ``P = 2**61 - 1`` and random integers ``1 <= p_i < P`` and ``0 <= q_i < P``: the pairwise-independent hash functions of the count-min sketch. The reference maps the key to ``(key * p_i + q_i) mod num_buckets`` in 32-bit integers. Without overflow, that bucket depends only on ``key mod num_buckets``, so every row would put the same edges in one bucket. The model takes the integer part of each node ID.

    MIDAS-R (Algorithm 2), which also scores the source and the destination nodes and decays the current counts instead of resetting them, is not implemented.

    Memory is fixed: two sketches of ``num_hash_fns`` by ``num_buckets`` counts.

    Args:
        num_hash_fns (int): The number of rows of each count-min sketch (Default=2, as in the paper).
        num_buckets (int): The number of buckets in each row (Default=2719, as in the paper).
        tick_size (int): The number of edges in each time tick (Default=1).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the hash functions. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self,
        num_hash_fns: int = 2,
        num_buckets: int = 2719,
        tick_size: int = 1,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.num_hash_fns = _int_at_least(num_hash_fns, "num_hash_fns", 1)
        self.num_buckets = _int_at_least(num_buckets, "num_buckets", 1)
        self.tick_size = _int_at_least(tick_size, "tick_size", 1)
        self.random_state = random_state
        self.num_fitted_ = 0

    def _buckets(self, X: np.ndarray) -> np.ndarray:
        """The bucket of the edge in each row, shape (num_hash_fns,)."""
        key = (int(X[0]) + _EDGE_KEY_PRIME * int(X[1])) % _HASH_PRIME
        # Python integers, since p * key overflows 64 bits.
        return np.array(
            [(p * key + q) % _HASH_PRIME % self.num_buckets for p, q in self.hash_params_]
        )

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> MIDAS:
        """Updates the edge counts with the next edge.

        Args:
            X (np.float64 array of shape (2,)): The edge to fit, as ``[source, destination]``.
            y (int): Ignored since the model is unsupervised (Default=None).

        Raises:
            ValueError: If the first edge does not have exactly two values.

        Returns:
            object: The model itself.
        """
        if self.num_fitted_ == 0:
            if X.shape[0] != 2:
                raise ValueError(
                    f"MIDAS takes each edge as [source, destination], got {X.shape[0]} values."
                )
            rng = check_random_state(self.random_state)
            self.hash_params_ = [
                (
                    int(rng.randint(1, _HASH_PRIME, dtype=np.int64)),
                    int(rng.randint(0, _HASH_PRIME, dtype=np.int64)),
                )
                for _ in range(self.num_hash_fns)
            ]
            self.current_ = np.zeros((self.num_hash_fns, self.num_buckets))
            self.total_ = np.zeros((self.num_hash_fns, self.num_buckets))
            self._rows = np.arange(self.num_hash_fns)
        elif self.num_fitted_ % self.tick_size == 0:
            # A new tick starts.
            self.current_.fill(0.0)

        buckets = self._buckets(X)
        self.current_[self._rows, buckets] += 1.0
        self.total_[self._rows, buckets] += 1.0
        self.num_fitted_ += 1

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the next edge by the counts it would have after the model fits it. This method does not change the model.

        Args:
            X (np.float64 array of shape (2,)): The edge to score, as ``[source, destination]``.

        Returns:
            float: The anomalousness score of the edge, at least 0.0. Higher scores represent more anomalous edges.
        """
        t = self.num_fitted_ // self.tick_size + 1
        if t == 1:
            return 0.0

        buckets = self._buckets(X)
        new_tick = self.num_fitted_ % self.tick_size == 0
        current = (0.0 if new_tick else self.current_[self._rows, buckets].min()) + 1.0
        total = self.total_[self._rows, buckets].min() + 1.0
        return float((current - total / t) ** 2 * t * t / (total * (t - 1)))
