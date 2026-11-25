from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from pysad import _kernels
from pysad.core.base_model import BaseModel


class _CoresetTreeNode:
    """A node of the coreset tree (paper section 4.1).

    A leaf stores the points in its cluster, their weights, and the representative sampled from them. An inner node stores no points; its cluster is the union of its children's clusters. ``size`` is the total weight of the points in the node's cluster, and ``cost`` is the weighted sum of squared distances of those points to the node's representative for a leaf, or the sum of the children's costs for an inner node.
    """

    __slots__ = ("cost", "left", "points", "rep", "right", "size", "weights")

    def __init__(
        self,
        points: NDArray[np.float64] | None,
        weights: NDArray[np.float64] | None,
        rep: NDArray[np.float64],
        size: float,
        cost: float,
    ) -> None:
        self.points = points
        self.weights = weights
        self.rep = rep
        self.size = size
        self.cost = cost
        self.left: _CoresetTreeNode | None = None
        self.right: _CoresetTreeNode | None = None


class StreamKMPlusPlus(BaseModel):
    """Distance to the nearest center of a StreamKM++ streaming coreset clustering :cite:`ackermann2012streamkmpp`.

    StreamKM++ maintains a small weighted coreset of the stream with the merge-and-reduce technique (paper section 5.1). Points accumulate in a buffer of ``coreset_size`` points. When the buffer is full, the coreset construction reduces it to a weighted coreset of ``coreset_size`` points, which enters level 0 of the merge-and-reduce levels. Whenever two coresets represent the same number of stream points, they merge into one set and the coreset construction reduces it back to ``coreset_size`` points, which moves up to the next level. Level ``i`` therefore holds a coreset of at most ``coreset_size`` points that represents ``2**i * coreset_size`` stream points, until the cap folds further merges into the top level. The number of levels is capped at ``ceil(log2(n_max))``; beyond ``n_max`` stream points, a further merge folds into the top level instead of adding a level, so the stored points stay bounded by ``(ceil(log2(n_max)) + 1) * coreset_size``.

    The coreset construction (paper sections 4.2-4.3) builds a coreset tree over the points. The first point is picked with probability proportional to its weight. Each next point is picked by descending the tree from the root, choosing each child with probability proportional to its cost, and sampling a point from the chosen leaf with probability proportional to its weight times its squared distance to the leaf representative. The leaf is then split into the points nearer to its old representative and the points nearer to the new point. Each sampled point is weighted by the total weight of the points in its leaf.

    After every instance, the detector reduces the union of the buffer and all level coresets to a final coreset of ``coreset_size`` points (paper section 5.1) and runs weighted k-means++ seeding followed by Lloyd iterations on it five times independently, keeping the clustering with the lowest weighted k-means cost (paper section 5). The score of an instance is its Euclidean distance to the nearest center, so instances far from every center score high.

    The paper computes the clustering from the size-m coreset at any point of time, so this detector recomputes the final coreset and the clustering after every instance and each instance can be scored while the stream runs. The recompute runs the same procedure the paper would run from the coreset at that time, at a higher per-instance cost. The remaining deviations from the paper are: the level cap above, where the paper lets the number of buckets grow with ``log2(n / m)``; a full buffer is reduced to a coreset before entering level 0, while the paper moves the ``m`` raw points into the first bucket, which only differs by aggregating exact duplicates since reducing ``m`` distinct points to ``m`` points returns the same points with unit weights; and the paper's construction is unweighted, while here weights enter the first pick, the leaf costs, the within-leaf sampling, and the leaf weights. Instances with non-finite values are not fitted, since they would poison the squared distances. Before any instance is fitted the score is 0.0, and with fewer points than ``num_clusters`` the centers are the points seen so far.

    Args:
        num_clusters (int): The number of centers k (Default=5).
        coreset_size (int): The number of points m in each coreset and in the buffer (Default=200).
        num_iterations (int): The number of Lloyd iterations for the centers (Default=10).
        n_max (int): The stream length the level cap is sized for; the number of merge-and-reduce levels is capped at ``ceil(log2(n_max))`` (Default=1000000).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the coreset sampling and the k-means++ seeding. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
        use_rust (bool): Use the optional Lloyd kernel when available. Otherwise, use NumPy. Floating-point rounding can change cluster assignments (Default=False).
    """

    def __init__(
        self,
        num_clusters: int = 5,
        coreset_size: int = 200,
        num_iterations: int = 10,
        n_max: int = 1_000_000,
        random_state: int | np.random.RandomState | None = None,
        use_rust: bool = False,
    ) -> None:
        if num_clusters < 1:
            raise ValueError("num_clusters must be positive.")
        if coreset_size < 1:
            raise ValueError("coreset_size must be positive.")
        if num_iterations < 1:
            raise ValueError("num_iterations must be positive.")
        if n_max < 1:
            raise ValueError("n_max must be positive.")

        self.num_clusters = num_clusters
        self.coreset_size = coreset_size
        self.num_iterations = num_iterations
        self.n_max = n_max
        self.random_state = random_state
        self.use_rust = use_rust
        self.rng_ = check_random_state(random_state)
        self.centers_: NDArray[np.float64] = np.empty(
            (0, 0)
        )  # the current centers, (num_clusters, num_features)
        self._buffer: NDArray[np.float64] = np.empty(
            (0, 0)
        )  # the points since the last reduce, (coreset_size, num_features)
        self._num_buffered = 0  # the number of points in the buffer
        self._max_levels = max(1, math.ceil(math.log2(n_max)))
        self._levels: list[tuple[NDArray[np.float64], NDArray[np.float64]] | None] = [
            None
        ] * self._max_levels  # one coreset per level, or None

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        if self.random_state is None:
            # The RNG draws from NumPy's global state, which is not pickled.
            state["rng_"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        if self.rng_ is None:
            self.rng_ = check_random_state(self.random_state)
        else:
            self.rng_ = check_random_state(self.rng_)

    def _coreset(
        self, points: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Reduces `points` with `weights` to a weighted coreset of at most `coreset_size` points.

        The coreset tree (paper sections 4.2-4.3) starts as one leaf holding all points, with a representative picked with probability proportional to its weight. Each next point is picked by descending the tree, choosing each child with probability proportional to its cost, and sampling a point from the chosen leaf with probability proportional to its weight times its squared distance to the leaf representative. The leaf then splits into the points nearer to its old representative and the points nearer to the new point, so the partition is the tree partition, not the global nearest-sample assignment. Each sampled point is weighted by the total weight of the points in its leaf. Sampling stops early when every remaining point coincides with a picked one.

        Args:
            points (np.float64 array of shape (num_points, num_features)): The points to reduce.
            weights (np.float64 array of shape (num_points,)): The weight of each point.

        Returns:
            tuple: The sampled points of shape (num_sampled, num_features) and their weights of shape (num_sampled,), with the weights summing to the sum of `weights`.
        """
        num_points = points.shape[0]
        num_sampled = min(self.coreset_size, num_points)
        if num_sampled == num_points and np.unique(points, axis=0).shape[0] == num_points:
            # Sampling every distinct point leaves one point per leaf, so each sampled point keeps
            # its own weight; the reduce is the identity.
            return points.copy(), weights.copy()
        total_weight = float(weights.sum())
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            probs: NDArray[np.float64] = np.full(num_points, 1.0 / num_points)
        else:
            probs = weights / total_weight
        first = int(self.rng_.choice(num_points, p=probs))
        first_dists = ((points - points[first]) ** 2).sum(axis=1)
        root = _CoresetTreeNode(
            points,
            weights,
            points[first],
            total_weight,
            float((weights * first_dists).sum()),
        )

        splits_left = num_sampled - 1
        while splits_left > 0:
            # Descend from the root, choosing each child with probability proportional to its cost.
            node = root
            path = [root]
            while node.left is not None and node.right is not None:
                left = node.left
                right = node.right
                if node.cost <= 0.0:
                    break
                if self.rng_.uniform() < left.cost / node.cost:
                    node = left
                else:
                    node = right
                path.append(node)
            if node.left is not None:
                break
            leaf = node
            # Sample a point from the leaf with probability proportional to its weight times its
            # squared distance to the leaf representative.
            leaf_points = leaf.points
            leaf_weights = leaf.weights
            if leaf_points is None or leaf_weights is None:
                break
            diff = leaf_points - leaf.rep
            leaf_dists = np.einsum("ij,ij->i", diff, diff)
            leaf_probs = leaf_weights * leaf_dists
            leaf_total = float(leaf_probs.sum())
            if not np.isfinite(leaf_total) or leaf_total <= 0.0:
                break
            leaf_probs = leaf_probs / leaf_total
            new_rep = leaf_points[int(self.rng_.choice(leaf_points.shape[0], p=leaf_probs))]
            # Split the leaf into the points nearer to its old representative and the rest. The
            # child costs reuse the distances already computed for the split.
            diff = leaf_points - new_rep
            new_dists = np.einsum("ij,ij->i", diff, diff)
            first_mask = leaf_dists < new_dists
            first_weights = leaf_weights[first_mask]
            second_weights = leaf_weights[~first_mask]
            leaf.left = _CoresetTreeNode(
                leaf_points[first_mask],
                first_weights,
                leaf.rep,
                float(first_weights.sum()),
                float((first_weights * leaf_dists[first_mask]).sum()),
            )
            leaf.right = _CoresetTreeNode(
                leaf_points[~first_mask],
                second_weights,
                new_rep,
                float(second_weights.sum()),
                float((second_weights * new_dists[~first_mask]).sum()),
            )
            leaf.points = None
            leaf.weights = None
            leaf.cost = leaf.left.cost + leaf.right.cost
            for ancestor in reversed(path[:-1]):
                ancestor.cost = ancestor.left.cost + ancestor.right.cost  # type: ignore[union-attr]
            splits_left -= 1

        leaves: list[_CoresetTreeNode] = []
        stack = [root]
        while stack:
            current = stack.pop()
            if current.left is not None and current.right is not None:
                stack.append(current.left)
                stack.append(current.right)
            else:
                leaves.append(current)
        sampled_points = np.array([leaf.rep for leaf in leaves])
        sampled_weights = np.array([leaf.size for leaf in leaves], dtype=np.float64)

        return sampled_points, sampled_weights

    def _insert(self, coreset: tuple[NDArray[np.float64], NDArray[np.float64]]) -> None:
        """Inserts `coreset` into the merge-and-reduce levels, merging and reducing upwards.

        The levels are a binary counter: level ``i`` holds a coreset representing ``2**i * coreset_size`` stream points. A merge that would need a level past the cap folds into the top level instead, so the number of levels never grows past ``ceil(log2(n_max))``.
        """
        level = 0
        current = coreset
        while True:
            if level >= self._max_levels:
                top = self._levels[-1]
                if top is None:
                    self._levels[-1] = current
                    return
                merged_points = np.concatenate([top[0], current[0]])
                merged_weights = np.concatenate([top[1], current[1]])
                self._levels[-1] = self._coreset(merged_points, merged_weights)
                return
            existing = self._levels[level]
            if existing is None:
                self._levels[level] = current
                return
            merged_points = np.concatenate([existing[0], current[0]])
            merged_weights = np.concatenate([existing[1], current[1]])
            self._levels[level] = None
            current = self._coreset(merged_points, merged_weights)
            level += 1

    def _union(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The buffer and all level coresets as one weighted point set."""
        points: list[NDArray[np.float64]] = (
            [self._buffer[: self._num_buffered]] if self._num_buffered else []
        )
        weights: list[NDArray[np.float64]] = (
            [np.ones(self._num_buffered)] if self._num_buffered else []
        )
        for level in self._levels:
            if level is not None:
                points.append(level[0])
                weights.append(level[1])
        if not points:
            return np.empty((0, 0)), np.empty(0)

        return np.concatenate(points), np.concatenate(weights)

    def _kmeans_once(
        self, points: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """One run of weighted k-means++ seeding followed by Lloyd iterations.

        Args:
            points (np.float64 array of shape (num_points, num_features)): The points to cluster.
            weights (np.float64 array of shape (num_points,)): The weight of each point.

        Returns:
            np.float64 array of shape (num_centers, num_features): The centers, with num_centers the smaller of `num_clusters` and the number of points.
        """
        num_points, num_features = points.shape
        num_centers = min(self.num_clusters, num_points)
        centers = np.repeat(points[:1], num_centers, axis=0)
        sq_norms = (points**2).sum(axis=1)
        min_dists = np.full(num_points, np.inf)  # squared distance to the nearest center
        for j in range(num_centers):
            if j == 0:
                # The reference picks the first center uniformly from non-dummy points
                # (paper section 5), so points with zero weight are excluded and the
                # rest are equally likely, instead of picking proportional to weight.
                non_dummy = weights > 0.0
                if not non_dummy.any():
                    non_dummy = np.ones(num_points, dtype=bool)
                probs = np.where(non_dummy, 1.0 / float(non_dummy.sum()), 0.0)
            else:
                probs = weights * min_dists
                total = probs.sum()
                if not np.isfinite(total) or total <= 0.0:
                    break
                probs = probs / total
            idx = int(self.rng_.choice(num_points, p=probs))
            centers[j] = points[idx]
            min_dists = np.minimum(
                min_dists,
                np.maximum(sq_norms - 2.0 * points @ points[idx] + sq_norms[idx], 0.0),
            )

        kernel = _kernels.streamkm_lloyd
        if self.use_rust and kernel is not None:
            return kernel(points, weights, centers, self.num_iterations)

        weighted_points = points * weights[:, None]
        for _ in range(self.num_iterations):
            dists = np.maximum(
                sq_norms[:, None] - 2.0 * points @ centers.T + (centers**2).sum(axis=1), 0.0
            )
            nearest = dists.argmin(axis=1)
            sums = np.zeros((num_centers, num_features))
            np.add.at(sums, nearest, weighted_points)
            counts = np.bincount(nearest, weights=weights, minlength=num_centers)
            new_centers = centers.copy()
            nonempty = counts > 0
            new_centers[nonempty] = sums[nonempty] / counts[nonempty, None]
            if np.array_equal(new_centers, centers):
                break
            centers = new_centers

        return centers

    def _weighted_cost(
        self,
        points: NDArray[np.float64],
        weights: NDArray[np.float64],
        centers: NDArray[np.float64],
    ) -> float:
        """The weighted k-means cost of `centers` on `points` with `weights`."""
        dists = np.maximum(
            (points**2).sum(axis=1)[:, None] - 2.0 * points @ centers.T + (centers**2).sum(axis=1),
            0.0,
        ).min(axis=1)
        return float((weights * dists).sum())

    def _kmeans(
        self, points: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Weighted k-means++ run five times independently, keeping the lowest-cost clustering.

        Args:
            points (np.float64 array of shape (num_points, num_features)): The points to cluster.
            weights (np.float64 array of shape (num_points,)): The weight of each point.

        Returns:
            np.float64 array of shape (num_centers, num_features): The centers of the best run, with num_centers the smaller of `num_clusters` and the number of points.
        """
        best_centers = None
        best_cost = np.inf
        for _ in range(5):
            centers = self._kmeans_once(points, weights)
            cost = self._weighted_cost(points, weights, centers)
            if cost < best_cost:
                best_centers = centers
                best_cost = cost

        return best_centers  # type: ignore[return-value]

    def fit_partial(self, X: NDArray[np.float64], y: int | None = None) -> StreamKMPlusPlus:
        """Updates the model with the next instance and recomputes the centers.

        The centers come from the final size-m coreset of the buffer and all level coresets, clustered with five k-means++ restarts (paper sections 5 and 5.1).

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        x = np.array(X, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(x)):  # Would poison the squared distances.
            return self
        if self._buffer.size == 0:
            self._buffer = np.empty((self.coreset_size, x.shape[0]))
            self.centers_ = np.empty((0, x.shape[0]))

        self._buffer[self._num_buffered] = x
        self._num_buffered += 1
        if self._num_buffered == self.coreset_size:
            self._insert(self._coreset(self._buffer, np.ones(self.coreset_size)))
            self._num_buffered = 0

        points, weights = self._union()
        coreset_points, coreset_weights = self._coreset(points, weights)
        self.centers_ = self._kmeans(coreset_points, coreset_weights)

        return self

    def score_partial(self, X: NDArray[np.float64]) -> float:
        """Scores the anomalousness of the next instance by its distance to the nearest center.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.centers_.shape[0] == 0:
            return 0.0

        x = np.asarray(X, dtype=np.float64)
        return float(np.linalg.norm(self.centers_ - x, axis=1).min())
