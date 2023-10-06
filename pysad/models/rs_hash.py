from pysad.core.base_model import BaseModel
import numpy as np


class RSHash(BaseModel):
    """Subspace outlier detection in linear time with randomized hashing :cite:`sathe2016subspace`. This implementation is adapted from `cmuxstream-baselines <https://github.com/cmuxstream/cmuxstream-baselines/blob/master/Dynamic/RS_Hash/sparse_stream_RSHash.py>`_.

        Args:
            feature_mins (np.float64 array of shape (num_features,)): Minimum boundary of the features.
            feature_maxes (np.float64 array of shape (num_features,)): Maximum boundary of the features.
            sampling_points (int): The number of sampling points (Default=1000).
            decay (float): The decay hyperparameter (Default=0.015).
            num_components (int): The number of ensemble components (Default=100).
            num_hash_fns (int): The number of hashing functions (Default=1).
    """

    def __init__(
            self,
            feature_mins,
            feature_maxes,
            sampling_points=1000,
            decay=0.015,
            num_components=100,
            num_hash_fns=1):
        self.minimum = feature_mins
        self.maximum = feature_maxes

        self.m = num_components
        self.w = num_hash_fns
        self.s = sampling_points
        self.dim = len(self.minimum)
        self.decay = decay
        self.scores = []
        self.num_hash = num_hash_fns
        self.cmsketches = []
        self.effS = max(1000, 1.0 / (1 - np.power(2, -self.decay)))

        self.f = np.random.uniform(
            low=1.0 / np.sqrt(self.effS), high=1 - (1.0 / np.sqrt(self.effS)), size=self.m)

        for i in range(self.num_hash):
            self.cmsketches.append({})

        self._sample_dims()

        self.alpha = self._sample_shifts()

        self.index = 0 + 1 - self.s

        self.last_score = None

    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: Returns the self.
        """
        score_instance = 0
        for r in range(self.m):
            Y = -1 * np.ones(len(self.V[r]))
            Y[range(len(self.V[r]))] = np.floor(
                (X[np.array(self.V[r])] + np.array(self.alpha[r])) / float(self.f[r]))

            mod_entry = np.insert(Y, 0, r)
            mod_entry = tuple(mod_entry.astype(np.int32))

            c = []
            for w in range(len(self.cmsketches)):
                try:
                    value = self.cmsketches[w][mod_entry]
                except KeyError:
                    value = (self.index, 0)

                # Scoring the Instance
                tstamp = value[0]
                wt = value[1]
                new_wt = wt * np.power(2, -self.decay * (self.index - tstamp))
                c.append(new_wt)

                # Update the instance
                new_tstamp = self.index
                self.cmsketches[w][mod_entry] = (new_tstamp, new_wt + 1)

            min_c = min(c)
            c = np.log(1 + min_c)
            score_instance = score_instance + c

        self.last_score = score_instance / self.m

        self.index += 1

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance. Outputs the last score. Note that this method must be called after the fit_partial

        Args:
            X (any): Ignored.
        Returns:
            float: The anomalousness score of the last fitted instance.
        """
        return self.last_score

    def _sample_shifts(self):
        alpha = []
        for r in range(self.m):
            alpha.append(
                np.random.uniform(
                    low=0,
                    high=self.f[r],
                    size=len(self.V[r])))

        return alpha

    def _sample_dims(self):
        max_term = np.max((2 * np.ones(self.f.size), list(1.0 / self.f)), axis=0)
        common_term = np.log(self.effS) / np.log(max_term)
        low_value = 1 + 0.5 * common_term
        high_value = common_term

        self.r = np.empty([self.m, ], dtype=int)
        self.V = []
        for i in range(self.m):
            if np.floor(low_value[i]) == np.floor(high_value[i]):
                self.r[i] = 1
            else:
                self.r[i] = min(
                    np.random.randint(
                        low=low_value[i],
                        high=high_value[i]),
                    self.dim)
            all_feats = np.array(list(range(self.dim)), dtype=np.int32)

            choice_feats = all_feats[np.where(self.minimum != self.maximum)]
            sel_V = np.random.choice(
                choice_feats, size=self.r[i], replace=False)
            self.V.append(sel_V)
