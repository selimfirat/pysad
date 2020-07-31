from models.base_model import BaseModel
import numpy as np

class RSHash(BaseModel):
    """
    fit_partial must be followed by score_partial directly on the same instance.
    Reference: Adapted from https://github.com/cmuxstream/cmuxstream-baselines/blob/master/Dynamic/RS_Hash/sparse_stream_RSHash.py. RS-Hash Paper
    """

    def __init__(self, feature_mins, feature_maxes, sampling_points=1000, decay=0.015, num_components= 00, num_hash_fns=1, **kwargs):
        super().__init__(**kwargs)
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
        self.effS = max(1000, 1.0/(1 - np.power(2, -self.decay)))

        self.f = np.random.uniform(low=1.0 / np.sqrt(self.effS), high=1 - (1.0 / np.sqrt(self.effS)), size=self.m)

        for i in range(self.num_hash):
            self.cmsketches.append({})

        self._sample_dims()

        self.alpha = self._sample_shifts()

        self.index = 0 + 1 - self.s

        self.last_score = None

    def fit_partial(self, x, y=None):

        score_instance = 0
        for r in range(self.m):
            Y = -1 * np.ones(len(self.V[r]))
            Y[range(len(self.V[r]))] = np.floor(
                (x[np.array(self.V[r])] + np.array(self.alpha[r])) / float(self.f[r]))

            mod_entry = np.insert(Y, 0, r)
            mod_entry = tuple(mod_entry.astype(np.int))

            c = []
            for w in range(len(self.cmsketches)):
                try:
                    value = self.cmsketches[w][mod_entry]
                except KeyError as e:
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

    def score_partial(self, x):

        return self.last_score

    def _sample_shifts(self):
        alpha = []
        for r in range(self.m):
            alpha.append(np.random.uniform(low=0, high=self.f[r], size=len(self.V[r])))

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
                self.r[i] = min(np.random.randint(low=low_value[i], high=high_value[i]), self.dim)
            all_feats = np.array(range(self.dim))
            choice_feats = all_feats[np.where(self.minimum[all_feats] != self.maximum[all_feats])]
            sel_V = np.random.choice(choice_feats, size=self.r[i], replace=False)
            self.V.append(sel_V)

    def score_update_instance(self, x, index):
        score_instance = 0
        for r in range(self.m):
            Y = -1 * np.ones(len(self.V[r]))
            Y[range(len(self.V[r]))] = np.floor(
                (x[np.array(self.V[r])] + np.array(self.alpha[r])) / float(self.f[r]))

            mod_entry = np.insert(Y, 0, r)
            mod_entry = tuple(mod_entry.astype(np.int))

            c = []
            for w in range(len(self.cmsketches)):
                try:
                    value = self.cmsketches[w][mod_entry]
                except KeyError as e:
                    value = (index, 0)

                # Scoring the Instance
                tstamp = value[0]
                wt = value[1]
                new_wt = wt * np.power(2, -self.decay * (index - tstamp))
                c.append(new_wt)

                # Update the instance
                new_tstamp = index
                self.cmsketches[w][mod_entry] = (new_tstamp, new_wt + 1)

            min_c = min(c)
            c = np.log(1 + min_c)
            score_instance = score_instance + c

        score = score_instance / self.m

        return score
