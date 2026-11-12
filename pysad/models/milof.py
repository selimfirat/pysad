from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

from pysad.models.incremental_lof import IncrementalLOF, _lof

# The number of weighted c-means iterations used to merge two summaries, I2 in the paper.
_MERGE_ITERATIONS = 10


class MiLOF(IncrementalLOF):
    """Scores each instance by its local outlier factor (LOF) in a bounded memory that summarizes older points :cite:`salehi2016fast`.

    MiLOF keeps `window_size` recent raw points and `num_summaries` cluster centers that summarize older points, so the memory stays bounded at `window_size + num_summaries` points instead of growing with the stream. Every time the window fills, the algorithm clusters the oldest half of the window into `num_summaries` centers with c-means, keeps the newest half intact, and deletes the oldest half. If a previous summary exists, the new centers are merged with it by weighted c-means, where the weight of a center is the number of raw points it represents. `score_partial` returns the LOF of the instance against the current raw points and cluster centers, without inserting it, so scoring does not change the model state. Until the memory holds more than `num_neighbors` points, the score is 0.

    Deviations from the paper: the paper stops the nearest-neighbor search as soon as a cluster center is among the first `num_neighbors - 1` neighbors, because the remaining neighbors lie in the same cluster. This model treats cluster centers as ordinary neighbors and does not implement that early stop. The paper stores the LOF of each cluster center, but this model only keeps its k-distance and mean reachability distance, which are all that scoring needs; the center LOF is not stored. The paper deletes blocks of obsolete records, while this model summarizes the oldest half of the window whenever the window fills, and the default `window_size` of 1000 and `num_summaries` of 10 are choices of this implementation.

    Args:
        num_neighbors (int): The number of nearest neighbors, k in the paper (Default=10).
        window_size (int): The number of recent raw points kept, b in the paper. Must be even and at least `2 * num_neighbors + 2` (Default=1000).
        num_summaries (int): The number of cluster centers summarizing older points, c in the paper. Must be between 1 and `window_size // 2` (Default=10).
        random_state (int, np.random.RandomState or None): The seed or random number generator for the c-means clustering. None draws from NumPy's global random state (Default=None).
    """

    def __init__(
        self,
        num_neighbors: int = 10,
        window_size: int = 1000,
        num_summaries: int = 10,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if window_size % 2 != 0:
            raise ValueError(f"window_size must be even, got {window_size}.")
        if window_size < 2 * num_neighbors + 2:
            raise ValueError(
                f"window_size must be at least 2 * num_neighbors + 2 = {2 * num_neighbors + 2}, got {window_size}."
            )
        half = window_size // 2
        if num_summaries < 1 or num_summaries > half:
            raise ValueError(
                f"num_summaries must be between 1 and window_size // 2 = {half}, got {num_summaries}."
            )
        super().__init__(num_neighbors=num_neighbors, window_size=window_size)
        self.num_summaries = num_summaries
        self.random_state = random_state
        # The generator of the c-means clustering. It is resolved once, so that the clustering of
        # each summary continues the same random stream.
        self._rng = check_random_state(random_state)
        # The cluster centers summarizing older points, and the number of raw points each center represents.
        self.centers_: np.ndarray | None = None
        self.center_weights_: np.ndarray | None = None
        self.center_k_dists_: np.ndarray | None = None
        self.center_reach_dists_: np.ndarray | None = None
        # The previous summary, used to merge with the new one.
        self._prev_centers: np.ndarray | None = None
        self._prev_weights: np.ndarray | None = None
        self._prev_k_dists: np.ndarray | None = None
        self._prev_reach_dists: np.ndarray | None = None
        # The number of completed summarization steps. The first step has no previous summary to merge.
        self._merge_step = 0

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> MiLOF:
        """Inserts the instance into the window, and summarizes the oldest half into cluster centers when the window is full.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        super().fit_partial(X)
        if self.num_instances_ == self.window_size:
            self._summarize_and_merge()
        return self

    def _summarize_and_merge(self) -> None:
        # Cluster the oldest half of the window into `num_summaries` centers, keep the newest half,
        # merge with the previous summary if there is one, then rebuild the neighbors of the kept points.
        half = self.window_size // 2
        old = self.X_[:half]  # the oldest half, in arrival order
        new = self.X_[half : self.window_size]  # the newest half, kept intact

        km = KMeans(
            n_clusters=self.num_summaries,
            random_state=self._rng,
            n_init=10,
        ).fit(old)
        labels = km.labels_
        centers = km.cluster_centers_
        weights = np.bincount(labels, minlength=self.num_summaries).astype(np.float64)

        k_dists = np.empty(self.num_summaries)
        reach_dists = np.empty(self.num_summaries)
        fallback_k = self.k_dists_[:half].mean()
        fallback_r = self.reach_dists_[:half].mean()
        for j in range(self.num_summaries):
            members = labels == j
            if members.any():
                k_dists[j] = self.k_dists_[:half][members].mean()
                reach_dists[j] = self.reach_dists_[:half][members].mean()
            else:
                # An empty cluster has no members; fall back to the mean over the oldest half.
                k_dists[j] = fallback_k
                reach_dists[j] = fallback_r

        if self._merge_step > 0:
            centers, weights, k_dists, reach_dists = self._merge(
                centers, weights, k_dists, reach_dists
            )

        self.X_[:half] = new
        self.num_instances_ = half
        self._next = half
        self._update_neighbors(np.arange(half), exclude=None)
        self._update_reach_dists(np.arange(half))
        # Recompute the k-nearest neighbors of the kept raw points against the kept raw points
        # and the cluster centers, so that a center can set the k-distance and reachability
        # distance of a kept point (Section 4.2.3 and Algorithm 2, line 23 of the paper).
        self._update_kept_points_with_centers(centers, k_dists)

        self.centers_ = centers
        self.center_weights_ = weights
        self.center_k_dists_ = k_dists
        self.center_reach_dists_ = reach_dists
        self._prev_centers = centers
        self._prev_weights = weights
        self._prev_k_dists = k_dists
        self._prev_reach_dists = reach_dists
        self._merge_step += 1

    def _update_kept_points_with_centers(
        self,
        centers: np.ndarray,
        center_k_dists: np.ndarray,
    ) -> None:
        # Recomputes the k nearest neighbors of the kept raw points against the kept raw points
        # and the cluster centers, so that a center can be the k-th neighbor and set the
        # k-distance and reachability distance of a kept point. The neighbor lists keep referring
        # to raw slots; only the k-distances and reachability distances are updated to reflect
        # centers as candidates.
        n = self.num_instances_
        raw = self.X_[:n]
        cand = np.vstack([raw, centers])  # (n + c, D)
        dists = cdist(raw, cand)  # (n, n + c)
        dists[np.arange(n), np.arange(n)] = np.inf  # a point is not its own neighbor
        neighbors = np.argpartition(dists, self.num_neighbors - 1, axis=1)[:, : self.num_neighbors]
        nd = np.take_along_axis(dists, neighbors, axis=1)  # (n, k)
        self.k_dists_[:n] = nd.max(axis=1)
        is_center = neighbors >= n
        k_dist_nb = np.where(
            is_center,
            center_k_dists[np.clip(neighbors - n, 0, centers.shape[0] - 1)],
            self.k_dists_[np.clip(neighbors, 0, n - 1)],
        )
        reach = np.maximum(k_dist_nb, nd)
        self.reach_dists_[:n] = reach.mean(axis=1)

    def _merge(
        self,
        centers: np.ndarray,
        weights: np.ndarray,
        k_dists: np.ndarray,
        reach_dists: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Merges the current and previous summaries with weighted c-means, minimizing the weighted
        # squared distance to the merged centers. The weight of a center is the number of raw points
        # it represents (Equation 10 and 11 of the paper).
        num = self.num_summaries
        prev_centers = self._prev_centers
        prev_weights = self._prev_weights
        prev_k_dists = self._prev_k_dists
        prev_reach_dists = self._prev_reach_dists
        assert prev_centers is not None
        assert prev_weights is not None
        assert prev_k_dists is not None
        assert prev_reach_dists is not None
        points = np.vstack([centers, prev_centers])  # (2c, D)
        w = np.concatenate([weights, prev_weights])  # (2c,)
        k_all = np.concatenate([k_dists, prev_k_dists])  # (2c,)
        r_all = np.concatenate([reach_dists, prev_reach_dists])  # (2c,)
        merged = centers.copy()
        for _ in range(_MERGE_ITERATIONS):
            dists = cdist(points, merged)  # (2c, c)
            assign = dists.argmin(axis=1)
            for j in range(num):
                mask = assign == j
                if mask.any():
                    merged[j] = (points[mask] * w[mask, None]).sum(axis=0) / w[mask].sum()
        dists = cdist(points, merged)
        assign = dists.argmin(axis=1)
        new_weights = np.empty(num)
        new_k_dists = np.empty(num)
        new_reach_dists = np.empty(num)
        for j in range(num):
            mask = assign == j
            wj = w[mask]
            if mask.any():
                new_weights[j] = wj.sum()
                new_k_dists[j] = (wj * k_all[mask]).sum() / wj.sum()
                new_reach_dists[j] = (wj * r_all[mask]).sum() / wj.sum()
            else:
                # Keep the current center if the merge assigns no point to it.
                new_weights[j] = weights[j]
                new_k_dists[j] = k_dists[j]
                new_reach_dists[j] = reach_dists[j]
        return merged, new_weights, new_k_dists, new_reach_dists

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the instance by its LOF against the current raw points and cluster centers, without inserting it.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The LOF of the instance. Higher scores represent more anomalous instances.
        """
        n = self.num_instances_
        centers = self.centers_
        c = 0 if centers is None else centers.shape[0]
        if n + c <= self.num_neighbors:
            return 0.0

        raw = self.X_[:n]
        if c == 0:
            cand = raw
        else:
            assert centers is not None
            cand = np.vstack([raw, centers])
        dists = cdist(np.atleast_2d(X), cand)[0]  # (n + c,)
        neighbors = np.argpartition(dists, self.num_neighbors - 1)[: self.num_neighbors]
        dist_n = dists[neighbors]  # (num_neighbors,)

        is_center = neighbors >= n
        raw_idx = neighbors[~is_center]
        center_idx = neighbors[is_center] - n

        center_k_dists = self.center_k_dists_
        center_reach_dists = self.center_reach_dists_
        reach = np.empty(self.num_neighbors)
        neighbor_reach = np.empty(self.num_neighbors)
        if raw_idx.size:
            reach[~is_center] = np.maximum(self.k_dists_[raw_idx], dist_n[~is_center])
            neighbor_reach[~is_center] = self.reach_dists_[raw_idx]
        if center_idx.size:
            assert center_k_dists is not None
            assert center_reach_dists is not None
            reach[is_center] = np.maximum(center_k_dists[center_idx], dist_n[is_center])
            neighbor_reach[is_center] = center_reach_dists[center_idx]

        return float(_lof(reach.mean(keepdims=True), neighbor_reach[None])[0])
