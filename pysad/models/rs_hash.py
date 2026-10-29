from __future__ import annotations

import math

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _finite_float, _int_at_least, check_feature_bounds


class RSHash(BaseModel):
    """Scores each instance by the time-decayed counts of its cells in randomized subspace grids (RS-Hash) :cite:`sathe2016subspace`.

    This implementation is adapted from `cmuxstream-baselines <https://github.com/cmuxstream/cmuxstream-baselines/blob/master/Dynamic/RS_Hash/sparse_stream_RSHash.py>`_ and follows the streaming variant (RS-Stream) of the paper. Instances are normalized with `feature_mins` and `feature_maxes`, and the score is the negated average of log2(1 + c) over the ensemble, where c is the time-decayed count of the instance's grid cell, so that higher scores are more anomalous.

    Grid cell counts are kept in a count-min sketch (:cite:`sathe2016subspace`, §II-A): `num_hash_fns` (w) pairwise-independent hash tables of `hash_range` (p) slots each, so the sketch's memory is O(w * p) and stays constant regardless of stream length, at the cost of hash collisions that can only overestimate a cell's count. Taking the minimum decayed count over the w tables reduces that overestimate. As in the paper's streaming variant (§III), all `num_components` components share one sketch, which holds about `num_components / (1 - 2**-decay)` live insertions (about 9,700 at the defaults); increase `hash_range` along with `num_components` or the decay window (smaller `decay`), or collisions make scores less accurate than exact counting. Unlike the paper, which pairs p = 10,000 with w = 4, the default `num_hash_fns=1` takes no minimum, so it does not reduce the overestimate; pass `num_hash_fns=4` for the paper's sketch.

    Args:
        feature_mins (np.float64 array of shape (num_features,)): Minimum boundary of the features. Integer arrays are converted to float.
        feature_maxes (np.float64 array of shape (num_features,)): Maximum boundary of the features, at least `feature_mins`. Integer arrays are converted to float.
        decay (float): The decay hyperparameter. Must be a positive finite real number (a NumPy number is accepted, but not a bool): `TypeError` is raised for other types and `ValueError` for non-positive, NaN or infinite values (Default=0.015).
        num_components (int): The number of ensemble components. Must be an int >= 1 (a NumPy integer is accepted, but not a bool): `TypeError` is raised for other types and `ValueError` for values below 1 (Default=100).
        num_hash_fns (int): The number w of pairwise-independent hash tables in the count-min sketch, whose smallest count is read (Default=1, which takes no minimum; the paper uses 4). Must be an int >= 1 (a NumPy integer is accepted, but not a bool): `TypeError` is raised for other types and `ValueError` for values below 1.
        hash_range (int): The number p of slots per hash table of the count-min sketch, which all `num_components` components share (Default=10000, the paper's p). It should grow with `num_components` and the decay window, since the sketch holds about `num_components / (1 - 2**-decay)` live insertions. Must be an int >= 1 (a NumPy integer is accepted, but not a bool): `TypeError` is raised for other types and `ValueError` for values below 1.
        random_state (int, np.random.RandomState or None): Seed or random number generator for the grid sizes, subspaces and shifts of the ensemble components, and for the sketch's hash parameters. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).

    Raises:
        TypeError: If `decay` is not a real number (or is a bool), or if `num_components`, `num_hash_fns` or `hash_range` is not an integer (or is a bool).
        ValueError: If `decay` is not positive and finite, if `num_components`, `num_hash_fns` or `hash_range` is below 1, or if `feature_mins` and `feature_maxes` are not finite 1-D arrays of the same length with every minimum at most its maximum.
    """

    def __init__(
        self,
        feature_mins: np.ndarray | list[float] | tuple[float, ...],
        feature_maxes: np.ndarray | list[float] | tuple[float, ...],
        decay: float = 0.015,
        num_components: int = 100,
        num_hash_fns: int = 1,
        hash_range: int = 10000,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        decay = _finite_float(decay, "decay")
        if decay <= 0.0 or 2.0**-decay == 1.0:
            raise ValueError(f"decay must be positive, got {decay}.")
        num_components = _int_at_least(num_components, "num_components", 1)
        # Stored as Python ints: a NumPy integer narrower than 64 bits would overflow in _cell_slots.
        num_hash_fns = _int_at_least(num_hash_fns, "num_hash_fns", 1)
        hash_range = _int_at_least(hash_range, "hash_range", 1)

        self.minimum, self.maximum = check_feature_bounds(feature_mins, feature_maxes)
        self.range = self.maximum - self.minimum
        self.range[self.range == 0] = 1.0

        self.m = num_components
        self.dim = len(self.minimum)
        self.decay = decay
        self.num_hash = num_hash_fns
        self.hash_range = hash_range
        self.effS = max(1000.0, 1.0 / (1.0 - 2.0**-self.decay))
        self.random_state = random_state
        rng = check_random_state(random_state)

        self.f = rng.uniform(
            low=1.0 / np.sqrt(self.effS), high=1 - (1.0 / np.sqrt(self.effS)), size=self.m
        )

        # Count-min sketch of fixed size (w=num_hash_fns tables, p=hash_range slots each), shared by
        # all components, so memory stays constant however long the stream runs. sketch_timestamps
        # holds the last index each slot was updated at; sketch_counts holds its decayed count as of
        # that index.
        self.sketch_timestamps = np.zeros((self.num_hash, self.hash_range), dtype=np.int64)
        self.sketch_counts = np.zeros((self.num_hash, self.hash_range), dtype=np.float64)

        self._sample_dims(rng)

        self.alpha = self._sample_shifts(rng)
        self._V_flat = np.concatenate(self.V)
        self._alpha_flat = np.concatenate(self.alpha)
        self._f_flat = np.repeat(self.f, self.r)
        ends = np.cumsum(self.r).tolist()
        self._component_slices: list[tuple[int, int]] = list(zip([0] + ends[:-1], ends, strict=True))

        # Pairwise-independent hash parameters, one (a_k, b_k) pair per table, drawn from rng
        # so that random_state makes the sketch's slot assignment reproducible.
        # P is a Mersenne prime. _cell_slots reduces hash(key), a signed 64-bit value that is often
        # larger than P in magnitude, mod P before computing ((a_k * h + b_k) mod P) mod p, all in
        # Python ints (not numpy int64) to avoid overflow. Drawn after
        # _sample_dims/_sample_shifts so that num_hash_fns cannot perturb the sampled subspaces
        # (self.V) or shifts (self.alpha): those draws must depend only on num_components and the
        # seed, not on how many hash tables the sketch happens to have. Each table's (a_k, b_k)
        # pair is drawn with its own pair of scalar calls, one table at a time (rather than one
        # vectorized call per parameter across all tables), so table k's parameters depend only on
        # the draws made for tables 0..k-1, not on num_hash_fns itself. dtype=np.int64 is required
        # because randint defaults to the C long, which is 32 bits on Windows and cannot hold P.
        self._prime = (1 << 61) - 1
        self._hash_params: list[tuple[int, int]] = []
        for _ in range(self.num_hash):
            a = int(rng.randint(1, self._prime, dtype=np.int64))
            b = int(rng.randint(0, self._prime, dtype=np.int64))
            self._hash_params.append((a, b))

        self.index = 1

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> RSHash:
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        self._fit_slots([self._cell_slots(key) for key in self._cell_keys(X)])

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance. This method does not change the model.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The anomalousness score of the input instance. Higher scores represent more anomalous instances.
        """
        return self._score_slots([self._cell_slots(key) for key in self._cell_keys(X)])

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Scores the next instance against the current state, then fits the model to it.

        The paper's streaming variant scores an instance before learning it (Sathe & Aggarwal 2016, §III): the "testing step" reads the current hash table, and the "training update" then updates the counts.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score and fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance.
        """
        slots = [self._cell_slots(key) for key in self._cell_keys(X)]
        score = self._score_slots(slots)
        self._fit_slots(slots)

        return score

    def _fit_slots(self, slots: list[list[int]]) -> None:
        """Updates the sketches at the given slot indices of an instance."""
        for row_slots in slots:
            for w, slot in enumerate(row_slots):
                decayed_wt = self._decayed_count(w, slot)

                self.sketch_timestamps[w, slot] = self.index
                self.sketch_counts[w, slot] = decayed_wt + 1.0

        self.index += 1

    def _score_slots(self, slots: list[list[int]]) -> float:
        """Scores an instance from its sketch slot indices, without writing to the sketches."""
        score_instance = 0.0
        for row_slots in slots:
            min_c = min(self._decayed_count(w, slot) for w, slot in enumerate(row_slots))
            score_instance += math.log2(1.0 + min_c)

        # Low counts indicate outliers in the paper, so the average log-count is negated to make higher scores more anomalous.
        return -score_instance / self.m

    def _cell_keys(self, X: np.ndarray) -> list[tuple[int, ...]]:
        """Computes each ensemble component's grid cell key for a normalized instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to hash.

        Returns:
            list of tuple of int: The cell key of each ensemble component, in order: the component's index followed by its zigzag-encoded grid coordinates.
        """
        # Equation 1 of the paper: normalize with the minimum and maximum of each feature.
        X = (np.asarray(X, dtype=np.float64) - self.minimum) / self.range

        # Convert floored float64 coordinates directly to Python ints instead of casting to a
        # fixed-width NumPy integer, which would saturate for far out-of-range instances.
        # Zigzag-encode the coordinates (0, -1, 1, -2, 2, ... -> 0, 1, 2, 3, 4, ...) so none is -1:
        # CPython hashes -1 like -2, so cells differing only in a -1/-2 coordinate would
        # otherwise hash alike and share a slot in every table.
        coords = [
            2 * y if (y := int(v)) >= 0 else -2 * y - 1
            for v in np.floor((X[self._V_flat] + self._alpha_flat) / self._f_flat).tolist()
        ]
        return [(r, *coords[start:end]) for r, (start, end) in enumerate(self._component_slices)]

    def _cell_slots(self, key: tuple[int, ...]) -> list[int]:
        """Maps a grid cell key to one slot per hash table of the count-min sketch.

        Uses w pairwise-independent hash functions of the form ``((a_k * h + b_k) mod P) mod p``,
        where h is hash(key) reduced mod P and shared by every table, P is the Mersenne prime
        2**61 - 1, and a_k, b_k are the per-table parameters drawn in `__init__`.

        Args:
            key (tuple of int): The grid cell key, as returned by `_cell_keys`.

        Returns:
            list of int: The slot index into each of the `num_hash` hash tables, in order.
        """
        h = hash(key) % self._prime
        return [((a * h + b) % self._prime) % self.hash_range for a, b in self._hash_params]

    def _decayed_count(self, w: int, slot: int) -> float:
        """Reads a hash table's time-decayed count for a sketch slot, without writing it back.

        Args:
            w (int): The index of the hash table.
            slot (int): The slot index within the table, as returned by `_cell_slots`.

        Returns:
            float: The count decayed to `self.index`, or 0 for a never-updated slot.
        """
        tstamp = int(self.sketch_timestamps[w, slot])
        wt = float(self.sketch_counts[w, slot])

        return wt * 2.0 ** (-self.decay * (self.index - tstamp))

    def _sample_shifts(self, rng: np.random.RandomState) -> list[np.ndarray]:
        alpha: list[np.ndarray] = []
        for r in range(self.m):
            alpha.append(rng.uniform(low=0, high=self.f[r], size=len(self.V[r])))

        return alpha

    def _sample_dims(self, rng: np.random.RandomState) -> None:
        # Dimensions with max == min are dropped from the candidate subspaces.
        all_feats = np.arange(self.dim)
        choice_feats = all_feats[self.minimum != self.maximum]
        if len(choice_feats) == 0:
            choice_feats = all_feats

        # r is an integer drawn uniformly between 1 + 0.5 * log_{max(2, 1/f)}(s) and log_{max(2, 1/f)}(s).
        max_term = np.maximum(2.0, 1.0 / self.f)
        common_term = np.log(self.effS) / np.log(max_term)
        high_value = np.floor(common_term).astype(int)
        low_value = np.minimum(np.ceil(1 + 0.5 * common_term).astype(int), high_value)

        self.r: np.ndarray = np.empty([self.m], dtype=int)
        self.V: list[np.ndarray] = []
        for i in range(self.m):
            self.r[i] = min(
                rng.randint(low=low_value[i], high=high_value[i] + 1), len(choice_feats)
            )
            self.V.append(rng.choice(choice_feats, size=self.r[i], replace=False))
