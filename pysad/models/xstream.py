from __future__ import annotations

from collections import Counter
from itertools import repeat
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.explainability import treeshap_ensemble
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
        self,
        num_components: int = 100,
        n_chains: int = 100,
        depth: int = 15,
        window_size: int = 256,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")
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
        self.first_window: list[NDArray[np.float64]] | None = []

    def _fit_projected(self, X: NDArray[np.float64], keys: list[bytes]) -> None:
        self.step += 1

        if self.first_window is not None:
            self.first_window.append(X)

        self.hs_chains.counts_cur.update(keys)

        # The windows swap after the last instance of a full window, so that the next instance
        # is scored against the window that just ended.
        if self.step % self.window_size == 0:
            if self.first_window is not None:
                self._set_bin_widths(np.concatenate(self.first_window, axis=0))
                self.first_window = None
            self.hs_chains.next_window()

    def fit_partial(self, X: NDArray[np.float64], y: int | None = None) -> xStream:
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        X = self.streamhash.fit_transform_partial(X).reshape(1, -1)
        self._fit_projected(X, self.hs_chains._bin_keys(X))

        return self

    def score_partial(self, X: NDArray[np.float64]) -> float:
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        X = self.streamhash.fit_transform_partial(X).reshape(1, -1)
        score = self.hs_chains.score(X).flatten()

        return float(score[0])

    def fit_score_partial(self, X: NDArray[np.float64], y: int | None = None) -> float:
        """Scores the next instance, and then updates the model with it.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance.
        """
        X = self.streamhash.fit_transform_partial(X).reshape(1, -1)
        keys = self.hs_chains._bin_keys(X)
        score = self.hs_chains._score_keys(keys).flatten()
        self._fit_projected(X, keys)

        return float(score[0])

    def _collect_chain_structure(
        self, chain_id: int, projected: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.intp],
        NDArray[np.float64],
        NDArray[np.intp],
        NDArray[np.intp],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Collect the structure of a single half-space chain for TreeSHAP.

        The chain is represented as a tree where at each depth d, the instance can either
        continue to the next depth (left branch) or stop at this depth (right branch).
        The right branch leaf has the score contribution at depth d.
        This allows TreeSHAP to attribute the score to features at different depths.

        Returns:
            Tuple of (features, thresholds, left, right, values, left_fractions) for the chain.
        """
        chains = self.hs_chains
        depth = chains.depth

        # Tree structure: each depth d has a split node.
        # Left child continues to depth d+1 (or final leaf at depth D).
        # Right child is a leaf with value = score contribution at depth d.
        # Total nodes: depth split nodes + (depth + 1) leaves = 2*depth + 1
        num_nodes = 2 * depth + 1
        features = np.full(num_nodes, -1, dtype=np.intp)
        thresholds = np.zeros(num_nodes, dtype=np.float64)
        left = np.full(num_nodes, -1, dtype=np.intp)
        right = np.full(num_nodes, -1, dtype=np.intp)
        values = np.zeros(num_nodes, dtype=np.float64)
        left_fractions = np.full(num_nodes, 0.5, dtype=np.float64)

        # Compute score contributions at each depth
        keys = chains._bin_keys(projected.reshape(1, -1))
        counts = np.fromiter(
            map(chains.counts.get, keys, repeat(0)), dtype=np.float64, count=len(keys)
        ).reshape(1, chains.nchains, chains.depth)
        chain_counts = counts[0, chain_id]
        score_contribs = -(np.log2(1.0 + chain_counts) + chains._depths)

        # Build the chain as a sequence of split nodes with right leaves
        for d in range(depth):
            node_idx = d
            feat = chains.fs[chain_id, d]
            features[node_idx] = feat
            shift = chains._shift_by_depth[chain_id, d]
            deltamax = chains._deltamax_by_depth[chain_id, d]
            bin_val = np.floor((projected[feat] + shift) / deltamax)
            # Threshold at the upper boundary of the instance's bin
            thresholds[node_idx] = (bin_val + 0.5) * deltamax - shift
            if d < depth - 1:
                left[node_idx] = d + 1  # Continue to next split node
            else:
                left[node_idx] = depth  # Final leaf (at index depth)
            right[node_idx] = depth + 1 + d  # Right leaf for stopping at depth d
            left_fractions[node_idx] = 0.5

        # Leaf nodes: index depth is the "continue all the way" leaf
        # Indices depth+1 to 2*depth are "stop at depth d" leaves
        values[depth] = score_contribs[depth - 1]  # Continue to max depth
        for d in range(depth):
            values[depth + 1 + d] = score_contribs[d]  # Stop at depth d

        return features, thresholds, left, right, values, left_fractions

    def score_explain_partial(self, X: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Returns the anomaly score and per-feature SHAP values for the next instance.

        Uses the TreeSHAP algorithm :cite:`lundberg2020consistent` to compute SHAP values
        for the half-space chain ensemble. Each chain is represented as a tree where
        each depth corresponds to a split on the chain's feature. The SHAP values explain
        the chain's anomaly score contribution (negative minimum of log2(count) + depth
        across depths), averaged across chains.
        Positive SHAP values indicate features that increase the anomaly score (more anomalous),
        negative values indicate features that decrease the anomaly score (more normal).

        Like ``score_partial``, it does not update the model or record the instance. Before
        the first window is processed, the method returns 0.0 and zeros of shape
        ``(num_components,)``.

        Note: The half-space chains are approximated as tree structures for TreeSHAP.
        The explanation operates in the projected (StreamHash) feature space.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score and explain.

        Returns:
            tuple: (score, shap_values) where
                score (float): The anomalousness score of the input instance.
                shap_values (np.float64 array of shape (num_components,)): SHAP values
                in the projected feature space, one per component. Positive values mean
                the component increases the anomaly score (more anomalous).

        Raises:
            ValueError: If `X` holds NaN.
        """
        x = np.asarray(X)
        if x.dtype.kind in "fc" and bool(np.isnan(x).any()):
            raise ValueError(f"{type(self).__name__} does not accept NaN values.")

        score = self.score_partial(X)

        if self.step == 0:
            return score, np.zeros(self.streamhash.n_components, dtype=np.float64)

        # Project the instance
        projected = self.streamhash.fit_transform_partial(x).reshape(1, -1)[0]

        # Collect tree structures for all chains
        trees_features: list[NDArray[np.intp]] = []
        trees_thresholds: list[NDArray[np.float64]] = []
        trees_left: list[NDArray[np.intp]] = []
        trees_right: list[NDArray[np.intp]] = []
        trees_values: list[NDArray[np.float64]] = []
        trees_left_fractions: list[NDArray[np.float64]] = []

        for chain_id in range(self.hs_chains.nchains):
            feat, thresh, left, right, vals, left_fracs = self._collect_chain_structure(
                chain_id, projected
            )
            trees_features.append(feat)
            trees_thresholds.append(thresh)
            trees_left.append(left)
            trees_right.append(right)
            trees_values.append(vals)
            trees_left_fractions.append(left_fracs)

        if not trees_features:
            return score, np.zeros(self.hs_chains.nchains, dtype=np.float64)

        shap_values = treeshap_ensemble(
            projected,
            trees_features,
            trees_thresholds,
            trees_left,
            trees_right,
            trees_values,
            len(projected),
            trees_left_fractions=trees_left_fractions,
        )

        # The SHAP values are in the projected space (num_components dimensions).
        # But we want to return them in the original feature space.
        # For now, return projected space SHAP values.
        return score, shap_values

    def _set_bin_widths(self, window: NDArray[np.float64]) -> None:
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

    def __init__(
        self,
        deltamax: NDArray[np.float64],
        n_chains: int = 100,
        depth: int = 15,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        k = len(deltamax)
        rng = check_random_state(random_state)

        self.nchains = n_chains
        self.depth = depth

        # Draw the split features and the shifts chain by chain to keep the order of the random numbers.
        self.fs: NDArray[np.intp] = np.empty((n_chains, depth), dtype=np.intp)
        self.rand_arr: NDArray[np.float64] = np.empty((n_chains, k))
        for c in range(n_chains):
            self.fs[c] = [rng.randint(0, k) for d in range(depth)]
            self.rand_arr[c] = rng.rand(k)

        self.set_deltamax(deltamax)

        # A split on a feature is stored in the slot of the first depth that splits on the same feature, so a bin only holds the features its chain splits on.
        self.slots: NDArray[np.intp] = np.argmax(self.fs[:, :, None] == self.fs[:, None, :], axis=2)
        self._flat_slots: NDArray[np.intp] = np.arange(n_chains)[:, None] * depth + self.slots
        self.first_split: NDArray[np.bool_] = self.slots == np.arange(depth)
        self.chain_depth_ids: NDArray[np.int32] = np.arange(
            n_chains * depth, dtype=np.int32
        ).reshape(n_chains, depth)
        self._depths: NDArray[np.intp] = np.arange(1, depth + 1)

        # In the first window, the reference and current counts are the same.
        self.counts: Counter[bytes] = Counter()
        self.counts_cur: Counter[bytes] = self.counts

    def _bin_keys(self, X: NDArray[np.float64]) -> list[bytes]:
        # Returns the bytes key of the bin of every instance, chain and depth (flattened in that order).
        n = X.shape[0]
        first = (X[:, self.fs] + self._shift_by_depth) / self._deltamax_by_depth
        prebins = np.zeros((n, self.nchains * self.depth), dtype=np.float64)
        cur_bins = np.zeros((n, self.nchains * self.depth), dtype=np.int32)
        bins = np.empty((n, self.nchains, self.depth, self.depth + 1), dtype=np.int32)
        bins[..., 0] = self.chain_depth_ids

        for depth in range(self.depth):
            slot = self._flat_slots[:, depth]
            updated = np.where(
                self.first_split[:, depth],
                first[:, :, depth],
                2.0 * prebins[:, slot] - self._shift_over_deltamax[:, depth],
            )
            prebins[:, slot] = updated
            cur_bins[:, slot] = np.floor(updated).astype(np.int32)
            bins[:, :, depth, 1:] = cur_bins.reshape(n, self.nchains, self.depth)

        return cast(
            list[bytes],
            bins.view(np.dtype((np.void, bins.shape[-1] * bins.itemsize))).reshape(-1).tolist(),
        )

    def _score_keys(self, keys: list[bytes], n: int = 1) -> NDArray[np.float64]:
        counts = np.fromiter(
            map(self.counts.get, keys, repeat(0)), dtype=np.float64, count=len(keys)
        ).reshape(n, self.nchains, self.depth)

        # scale score logarithmically to avoid overflow:
        #    score = min_d [ log2(bincount x 2^d) = log2(bincount) + d ]
        scores = -np.min(np.log2(1.0 + counts) + self._depths, axis=2)  # add 1 to avoid log(0)

        # Sum the chains sequentially, as a pairwise sum would round differently.
        scores = np.add.accumulate(scores, axis=1)[:, -1]
        scores /= float(self.nchains)
        return scores

    def score(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return self._score_keys(self._bin_keys(X), len(X))

    def next_window(self) -> None:
        self.counts = self.counts_cur
        self.counts_cur = Counter()

    def set_deltamax(self, deltamax: NDArray[np.float64]) -> None:
        self.deltamax = deltamax
        self.shift = self.rand_arr * deltamax
        self._shift_by_depth = self.shift[np.arange(self.nchains)[:, None], self.fs]
        self._deltamax_by_depth = self.deltamax[self.fs]
        self._shift_over_deltamax = self._shift_by_depth / self._deltamax_by_depth
