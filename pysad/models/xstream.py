from collections import Counter
from itertools import repeat

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.transform.projection.streamhash_projector import StreamhashProjector
from pysad.utils import get_minmax_array


class xStream(BaseModel):
    """Scores each instance by its density in half-space chains over StreamHash projections (xStream) :cite:`xstream`.

    It projects each instance with StreamHash and bins it in half-space chains. The chains count the instances of the current window and score each instance with the counts of the previous (reference) window, and the two windows swap when the instance after a full window arrives, before it is counted (Algorithm 1 of the paper). As in the paper and the authors' code, the initial bin widths (half the range of each projected feature) and the shifts are estimated once from the first window and kept for the rest of the stream, so every window is binned the same way. Until the first window is full, the instances are binned with width 0.5 and scored with the counts of the first window so far. When it is full, the first window is counted again with the estimated widths. The defaults are 100 projections, 100 chains, depth 15 and a window of 256, as in the code of the authors. The paper uses a window of 25 in its experiments on evolving streams. For a short stream of a few thousand instances or fewer, a window of 25 can give better scores.

    Args:
        num_components (int): The number of components for streamhash projection (Default=100).
        n_chains (int): The number of half-space chains (Default=100).
        depth (int): The maximum depth for the chains (Default=15).
        window_size (int): The size (and the sliding length) of the reference window. For a short stream of a few thousand instances or fewer, use 25 (Default=256).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the split features and shifts of the chains. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self, num_components=100, n_chains=100, depth=15, window_size=256, random_state=None
    ):
        self.streamhash = StreamhashProjector(num_components=num_components)
        self.window_size = window_size
        self.random_state = random_state
        # The bin widths are estimated from the first window once it is full.
        self.hs_chains = _HSChains(
            deltamax=np.full(num_components, 0.5),
            n_chains=n_chains,
            depth=depth,
            random_state=random_state,
        )

        self.step = 0
        self.first_window = []

    def fit_partial(self, X, y=None):
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        self.step += 1

        X = self.streamhash.fit_transform_partial(X)

        X = X.reshape(1, -1)
        if self.first_window is not None:
            self.first_window.append(X)

        self.hs_chains.fit(X)

        # The windows swap after the last instance of a full window, so that the next instance
        # is scored against the window that just ended.
        if self.step % self.window_size == 0:
            if self.first_window is not None:
                self._set_bin_widths(np.concatenate(self.first_window, axis=0))
                self.first_window = None
            self.hs_chains.next_window()

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        X = self.streamhash.fit_transform_partial(X)
        X = X.reshape(1, -1)
        score = self.hs_chains.score(X).flatten()

        return score

    def _set_bin_widths(self, window):
        # Sets the bin widths to half the range of the first window and counts the first window again with them.
        mn, mx = get_minmax_array(window)
        deltamax = (mx - mn) / 2.0
        deltamax[np.abs(deltamax) <= 0.0001] = 1.0

        self.hs_chains.set_deltamax(deltamax)
        self.hs_chains.counts_cur = Counter(self.hs_chains._bin_keys(window))


class _HSChains:
    """Half-space chains that are fitted and scored together, vectorized across the chains.

    At depth `d`, a chain assigns an instance to the bin given by the floored values of the (shifted and repeatedly halved) features it split on up to `d`. The bin counts of all chains and depths are kept in a single `Counter` keyed by the bytes of `(chain * depth + d, bin)`.
    """

    def __init__(self, deltamax, n_chains=100, depth=15, random_state=None):
        k = len(deltamax)
        rng = check_random_state(random_state)

        self.nchains = n_chains
        self.depth = depth

        # Draw the split features and the shifts chain by chain to keep the order of the random numbers.
        self.fs = np.empty((n_chains, depth), dtype=np.intp)
        self.rand_arr = np.empty((n_chains, k))
        for c in range(n_chains):
            self.fs[c] = [rng.randint(0, k) for d in range(depth)]
            self.rand_arr[c] = rng.rand(k)

        self.set_deltamax(deltamax)

        # A split on a feature is stored in the slot of the first depth that splits on the same feature, so a bin only holds the features its chain splits on.
        self.slots = np.argmax(self.fs[:, :, None] == self.fs[:, None, :], axis=2)
        self.first_split = self.slots == np.arange(depth)
        self.chain_depth_ids = np.arange(n_chains * depth, dtype=np.int32).reshape(n_chains, depth)

        # In the first window, the reference and current counts are the same.
        self.counts = Counter()
        self.counts_cur = self.counts

    def _bin_keys(self, X):
        # Returns the bytes key of the bin of every instance, chain and depth (flattened in that order).
        n = X.shape[0]
        chains = np.arange(self.nchains)
        prebins = np.zeros((n, self.nchains, self.depth), dtype=np.float64)
        bins = np.empty((n, self.nchains, self.depth, self.depth + 1), dtype=np.int32)
        bins[..., 0] = self.chain_depth_ids

        for depth in range(self.depth):
            f = self.fs[:, depth]
            slot = self.slots[:, depth]
            shift = self.shift[chains, f]
            deltamax = self.deltamax[f]

            first = (X[:, f] + shift) / deltamax
            halved = 2.0 * prebins[:, chains, slot] - shift / deltamax
            prebins[:, chains, slot] = np.where(self.first_split[:, depth], first, halved)

            bins[:, :, depth, 1:] = np.floor(prebins).astype(np.int32)

        return bins.view(np.dtype((np.void, bins.shape[-1] * bins.itemsize))).reshape(-1).tolist()

    def _bin_counts(self, X):
        # Returns the reference count of the bin of every instance, chain and depth, of shape (n, nchains, depth).
        keys = self._bin_keys(X)
        counts = np.fromiter(
            map(self.counts.get, keys, repeat(0)), dtype=np.float64, count=len(keys)
        )
        return counts.reshape(X.shape[0], self.nchains, self.depth)

    def score(self, X):
        counts = self._bin_counts(X)

        # scale score logarithmically to avoid overflow:
        #    score = min_d [ log2(bincount x 2^d) = log2(bincount) + d ]
        depths = np.arange(1, self.depth + 1)
        scores = -np.min(np.log2(1.0 + counts) + depths, axis=2)  # add 1 to avoid log(0)

        # Sum the chains sequentially, as a pairwise sum would round differently.
        scores = np.add.accumulate(scores, axis=1)[:, -1]
        scores /= float(self.nchains)
        return scores

    def fit(self, X):
        self.counts_cur.update(self._bin_keys(X))

    def next_window(self):
        self.counts = self.counts_cur
        self.counts_cur = Counter()

    def set_deltamax(self, deltamax):
        self.deltamax = deltamax
        self.shift = self.rand_arr * deltamax
