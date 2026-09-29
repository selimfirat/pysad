import numbers
import warnings

import numpy as np

from pysad.core.base_model import BaseModel

_UNSET = object()


def _positive_int(value, name):
    """Returns `value` as a Python int, or raises if it is not an integer >= 1 (NumPy integers are accepted, bools are not)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {value!r}.")

    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}.")

    return value


class RSHash(BaseModel):
    """Subspace outlier detection in linear time with randomized hashing :cite:`sathe2016subspace`. This implementation is adapted from `cmuxstream-baselines <https://github.com/cmuxstream/cmuxstream-baselines/blob/master/Dynamic/RS_Hash/sparse_stream_RSHash.py>`_ and follows the streaming variant (RS-Stream) of the paper. Instances are normalized with `feature_mins` and `feature_maxes`, and the score is the negated average of log2(1 + c) over the ensemble, where c is the time-decayed count of the instance's grid cell, so that higher scores are more anomalous.

    Grid cell counts are kept in a count-min sketch (:cite:`sathe2016subspace`, §II-A): `num_hash_fns` (w) pairwise-independent hash tables of `hash_range` (p) slots each, so the sketch's memory is O(w * p) and stays constant regardless of stream length, at the cost of hash collisions that can only overestimate a cell's count. Taking the minimum decayed count over the w tables reduces that overestimate. As in the paper's streaming variant (§III), all `num_components` components share one sketch, which holds about `num_components / (1 - 2**-decay)` live insertions (about 9,700 at the defaults); increase `hash_range` along with `num_components` or the decay window (smaller `decay`), or collisions make scores less accurate than exact counting. Unlike the paper, which pairs p = 10,000 with w = 4, the default `num_hash_fns=1` takes no minimum, so it does not reduce the overestimate; pass `num_hash_fns=4` for the paper's sketch.

    Args:
        feature_mins (np.float64 array of shape (num_features,)): Minimum boundary of the features.
        feature_maxes (np.float64 array of shape (num_features,)): Maximum boundary of the features.
        sampling_points (int): Deprecated. Has no effect.
        decay (float): The decay hyperparameter (Default=0.015).
        num_components (int): The number of ensemble components (Default=100).
        num_hash_fns (int): The number w of pairwise-independent hash tables in the count-min sketch, whose smallest count is read (Default=1, which takes no minimum; the paper uses 4). Must be an int >= 1 (a NumPy integer is accepted, but not a bool): `TypeError` is raised for other types and `ValueError` for values below 1.
        hash_range (int): The number p of slots per hash table of the count-min sketch, which all `num_components` components share (Default=10000, the paper's p). It should grow with `num_components` and the decay window, since the sketch holds about `num_components / (1 - 2**-decay)` live insertions. Must be an int >= 1 (a NumPy integer is accepted, but not a bool): `TypeError` is raised for other types and `ValueError` for values below 1.

    .. deprecated:: 0.6.1
        The ``sampling_points`` parameter is deprecated and has no effect.
        It will be removed in a future release.
    """

    def __init__(
        self,
        feature_mins,
        feature_maxes,
        sampling_points=_UNSET,
        decay=0.015,
        num_components=100,
        num_hash_fns=1,
        hash_range=10000,
    ):
        # Stored as Python ints: a NumPy integer narrower than 64 bits would overflow in _cell_slots.
        num_hash_fns = _positive_int(num_hash_fns, "num_hash_fns")
        hash_range = _positive_int(hash_range, "hash_range")

        if sampling_points is not _UNSET:
            warnings.warn(
                "The 'sampling_points' parameter is deprecated and has no "
                "effect. It will be removed in a future release.",
                FutureWarning,
                stacklevel=2,
            )

        self.minimum = np.asarray(feature_mins, dtype=np.float64)
        self.maximum = np.asarray(feature_maxes, dtype=np.float64)
        self.range = self.maximum - self.minimum
        self.range[self.range == 0] = 1.0

        self.m = num_components
        self.dim = len(self.minimum)
        self.decay = decay
        self.scores = []
        self.num_hash = num_hash_fns
        self.hash_range = hash_range
        self.effS = max(1000, 1.0 / (1 - np.power(2, -self.decay)))

        self.f = np.random.uniform(
            low=1.0 / np.sqrt(self.effS), high=1 - (1.0 / np.sqrt(self.effS)), size=self.m
        )

        # Count-min sketch of fixed size (w=num_hash_fns tables, p=hash_range slots each), shared by
        # all components, so memory stays constant however long the stream runs. sketch_timestamps
        # holds the last index each slot was updated at; sketch_counts holds its decayed count as of
        # that index.
        self.sketch_timestamps = np.zeros((self.num_hash, self.hash_range), dtype=np.int64)
        self.sketch_counts = np.zeros((self.num_hash, self.hash_range), dtype=np.float64)

        self._sample_dims()

        self.alpha = self._sample_shifts()

        # Pairwise-independent hash parameters, one (a_k, b_k) pair per table, drawn from
        # np.random so that pysad.utils.fix_seed makes the sketch's slot assignment reproducible.
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
        self._hash_params = []
        for _ in range(self.num_hash):
            a = int(np.random.randint(1, self._prime, dtype=np.int64))
            b = int(np.random.randint(0, self._prime, dtype=np.int64))
            self._hash_params.append((a, b))

        self.index = 1

    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: Returns the self.
        """
        self._fit_keys(self._cell_keys(X))

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance. This method does not change the model.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The anomalousness score of the input instance. Higher scores represent more anomalous instances.
        """
        return self._score_keys(self._cell_keys(X))

    def fit_score_partial(self, X, y=None):
        """Scores the next instance against the current state, then fits the model to it.

        The paper's streaming variant scores an instance before learning it (Sathe & Aggarwal 2016, §III): the "testing step" reads the current hash table, and the "training update" then updates the counts.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score and fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance.
        """
        keys = self._cell_keys(X)
        score = self._score_keys(keys)
        self._fit_keys(keys)

        return score

    def _fit_keys(self, keys):
        """Updates the sketches with the cell keys of an instance.

        Args:
            keys (list of tuple): The cell keys of the instance, as returned by `_cell_keys`.
        """
        for mod_entry in keys:
            for w, slot in enumerate(self._cell_slots(mod_entry)):
                decayed_wt = self._decayed_count(w, slot)

                self.sketch_timestamps[w, slot] = self.index
                self.sketch_counts[w, slot] = decayed_wt + 1

        self.index += 1

    def _score_keys(self, keys):
        """Scores an instance from its cell keys, without writing to the sketches.

        Args:
            keys (list of tuple): The cell keys of the instance, as returned by `_cell_keys`.

        Returns:
            float: The anomalousness score of the input instance. Higher scores represent more anomalous instances.
        """
        score_instance = 0
        for mod_entry in keys:
            c = [self._decayed_count(w, slot) for w, slot in enumerate(self._cell_slots(mod_entry))]

            min_c = min(c)
            score_instance = score_instance + np.log2(1 + min_c)

        # Low counts indicate outliers in the paper, so the average log-count is negated to make higher scores more anomalous.
        return -score_instance / self.m

    def _cell_keys(self, X):
        """Computes each ensemble component's grid cell key for a normalized instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to hash.

        Returns:
            list of tuple of int: The cell key of each ensemble component, in order: the component's index followed by its zigzag-encoded grid coordinates.
        """
        # Equation 1 of the paper: normalize with the minimum and maximum of each feature.
        X = (np.asarray(X, dtype=np.float64) - self.minimum) / self.range

        mod_entries = []
        for r in range(self.m):
            Y = np.floor((X[self.V[r]] + self.alpha[r]) / float(self.f[r])).astype(np.int32)

            # Zigzag-encode the coordinates (0, -1, 1, -2, 2, ... -> 0, 1, 2, 3, 4, ...) so none is -1:
            # CPython hashes -1 like -2, so cells differing only in a -1/-2 coordinate would
            # otherwise hash alike and share a slot in every table. tolist() yields Python ints, so
            # the key hashes the same in every process.
            mod_entries.append((r,) + tuple(2 * y if y >= 0 else -2 * y - 1 for y in Y.tolist()))

        return mod_entries

    def _cell_slots(self, key):
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

    def _decayed_count(self, w, slot):
        """Reads a hash table's time-decayed count for a sketch slot, without writing it back.

        Args:
            w (int): The index of the hash table.
            slot (int): The slot index within the table, as returned by `_cell_slots`.

        Returns:
            float: The count decayed to `self.index`, or 0 for a never-updated slot.
        """
        tstamp = self.sketch_timestamps[w, slot]
        wt = self.sketch_counts[w, slot]

        return wt * np.power(2, -self.decay * (self.index - tstamp))

    def _sample_shifts(self):
        alpha = []
        for r in range(self.m):
            alpha.append(np.random.uniform(low=0, high=self.f[r], size=len(self.V[r])))

        return alpha

    def _sample_dims(self):
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

        self.r = np.empty([self.m], dtype=int)
        self.V = []
        for i in range(self.m):
            self.r[i] = min(
                np.random.randint(low=low_value[i], high=high_value[i] + 1), len(choice_feats)
            )
            self.V.append(np.random.choice(choice_feats, size=self.r[i], replace=False))
