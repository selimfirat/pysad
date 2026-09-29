import warnings

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel

_UNSET = object()


class RSHash(BaseModel):
    """Subspace outlier detection in linear time with randomized hashing :cite:`sathe2016subspace`. This implementation is adapted from `cmuxstream-baselines <https://github.com/cmuxstream/cmuxstream-baselines/blob/master/Dynamic/RS_Hash/sparse_stream_RSHash.py>`_ and follows the streaming variant (RS-Stream) of the paper. Instances are normalized with `feature_mins` and `feature_maxes`, and the score is the negated average of log2(1 + c) over the ensemble, where c is the time-decayed count of the instance's grid cell, so that higher scores are more anomalous.

    Args:
        feature_mins (np.float64 array of shape (num_features,)): Minimum boundary of the features.
        feature_maxes (np.float64 array of shape (num_features,)): Maximum boundary of the features.
        sampling_points (int): Deprecated. Has no effect.
        decay (float): The decay hyperparameter (Default=0.015).
        num_components (int): The number of ensemble components (Default=100).
        num_hash_fns (int): The number of hashing functions (Default=1).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the grid sizes, subspaces and shifts of the ensemble components. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).

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
        random_state=None,
    ):
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
        self.cmsketches = []
        self.effS = max(1000, 1.0 / (1 - np.power(2, -self.decay)))
        self.random_state = random_state
        rng = check_random_state(random_state)

        self.f = rng.uniform(
            low=1.0 / np.sqrt(self.effS), high=1 - (1.0 / np.sqrt(self.effS)), size=self.m
        )

        for _ in range(self.num_hash):
            self.cmsketches.append({})

        self._sample_dims(rng)

        self.alpha = self._sample_shifts(rng)

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
            for w in range(len(self.cmsketches)):
                decayed_wt = self._decayed_count(w, mod_entry)

                self.cmsketches[w][mod_entry] = (self.index, decayed_wt + 1)

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
            c = [self._decayed_count(w, mod_entry) for w in range(len(self.cmsketches))]

            min_c = min(c)
            score_instance = score_instance + np.log2(1 + min_c)

        # Low counts indicate outliers in the paper, so the average log-count is negated to make higher scores more anomalous.
        return -score_instance / self.m

    def _cell_keys(self, X):
        """Computes each ensemble component's grid cell key for a normalized instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to hash.

        Returns:
            list of tuple: The cell key of each ensemble component, in order.
        """
        # Equation 1 of the paper: normalize with the minimum and maximum of each feature.
        X = (np.asarray(X, dtype=np.float64) - self.minimum) / self.range

        mod_entries = []
        for r in range(self.m):
            Y = np.floor((X[self.V[r]] + self.alpha[r]) / float(self.f[r]))

            mod_entry = np.insert(Y, 0, r)
            mod_entries.append(tuple(mod_entry.astype(np.int32)))

        return mod_entries

    def _decayed_count(self, w, mod_entry):
        """Reads a hash function's time-decayed count for a grid cell, without writing it back.

        Args:
            w (int): The index of the hash function's sketch.
            mod_entry (tuple): The grid cell key, as returned by `_cell_keys`.

        Returns:
            float: The count decayed to `self.index`, or 0 for an unseen cell.
        """
        try:
            value = self.cmsketches[w][mod_entry]
        except KeyError:
            value = (self.index, 0)

        tstamp = value[0]
        wt = value[1]

        return wt * np.power(2, -self.decay * (self.index - tstamp))

    def _sample_shifts(self, rng):
        alpha = []
        for r in range(self.m):
            alpha.append(rng.uniform(low=0, high=self.f[r], size=len(self.V[r])))

        return alpha

    def _sample_dims(self, rng):
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
                rng.randint(low=low_value[i], high=high_value[i] + 1), len(choice_feats)
            )
            self.V.append(rng.choice(choice_feats, size=self.r[i], replace=False))
