from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _int_at_least


class RSForest(BaseModel):
    """Scores each instance by its density under a forest of random space trees, built from the previous window :cite:`wu2014rsforest`.

    Each random space tree (RS-Tree) is a full binary tree of depth ``max_depth``. Every internal node splits a randomly chosen feature at a uniformly random point of the node's range (Algorithm 1), so a node at the end of the splits ``r_1, ..., r_h`` covers the fraction ``prod r_k`` of the feature space (Lemma 2.2). A node profile is the number of instances of a window that fall into the node. The density of ``x`` in one tree is ``|l| / (N * V(l))`` (Definition 2.3), where ``l`` is the termination node of ``x``: the first node on its path whose profile is at most ``node_size_limit``, or the leaf. The forest averages the densities of its trees (Definition 2.4), and the score is the negative of this average, so higher scores are more anomalous. As in Equation 5, the volumes are relative to the volume of the estimated feature space, and ``N`` is ``window_size``.

    The model follows Algorithm 3. It keeps the first ``window_size`` instances to estimate the range of each feature, builds the trees, and fills the reference profile with these instances. Each later instance is scored against the reference profile and recorded into the latest profile. When ``window_size`` instances were recorded, the latest profile becomes the reference profile and the other profile is emptied (dual node profiles). The range of a feature is its 90% highest posterior density (HPD) interval, the shortest interval that holds 90% of its values in the first window (the estimator of Chen et al. that the paper uses), widened by three standard deviations on each side (Section II-B).

    The paper is one-class semi-supervised: the model gets the label of each instance after it scores it, and an anomaly does not enter the profiles (Algorithm 5). ``fit_partial`` takes this label as ``y``. If ``y`` is 1, the instance counts toward the window but not toward the profile. If ``y`` is None, every instance counts as normal, so the model is unsupervised.

    The implementation differs from the paper as follows:

    - Algorithm 5 records an instance only down to its termination node in the reference profile, and extends the path at the end of the window. The model records every instance along its whole path from the root to the leaf, which gives the node profiles of Definition 2.2 exactly. The paper gives no node size limit as the default. The model uses 10, which is in the best group of Figure 7b (window size 512).
    - The paper does not score the instances that build the trees. The model gives them the score 0.0.
    - The paper calls the HPD interval an interval for the mean of the feature. Its analysis in Section III-A uses ``mu ± 1.645 sigma`` for a normal feature, which is the interval of the values. The model follows the analysis.

    Memory is fixed: two node profiles, the split features and points, and one volume per node for each of ``num_trees`` trees of depth ``max_depth``, about 32 MB with the defaults. The first window is kept until the trees are built.

    Args:
        window_size (int): The number of instances in a window, ``psi`` in the paper. The first window estimates the feature ranges, and each window gives the reference profile for the next one (Default=512, as in the paper).
        num_trees (int): The number of RS-Trees, ``M`` in the paper (Default=30, as in the paper).
        max_depth (int): The depth of each RS-Tree, ``H`` in the paper (Default=15, as in the paper).
        node_size_limit (int): The node size limit ``zeta``. The density of a tree comes from the first node on the path of an instance whose profile is at most this limit, or from the leaf (Default=10).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the split features and points. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self,
        window_size: int = 512,
        num_trees: int = 30,
        max_depth: int = 15,
        node_size_limit: int = 10,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.window_size = _int_at_least(window_size, "window_size", 1)
        self.num_trees = _int_at_least(num_trees, "num_trees", 1)
        self.max_depth = _int_at_least(max_depth, "max_depth", 1)
        self.node_size_limit = _int_at_least(node_size_limit, "node_size_limit", 0)
        self.random_state = random_state

        num_splits = 2**self.max_depth - 1
        num_nodes = 2 * num_splits + 1
        # Trees are in heap order: node i has the children 2i + 1 and 2i + 2. The arrays of all trees are
        # read flat, where node i of tree t sits at t * num_splits + i (splits) or t * num_nodes + i (nodes).
        self._split_offsets = np.arange(self.num_trees) * num_splits
        self._node_offsets = np.arange(self.num_trees) * num_nodes
        self._trees = np.arange(self.num_trees)

        self.initial_window_: np.ndarray | None = None
        self.num_initial_ = 0
        self.num_recorded_ = 0

    def _build(self, X: np.ndarray) -> None:
        """Builds the trees from the first window `X` of shape (window_size, num_features) and fills the reference profile with it."""
        num_instances, num_features = X.shape
        # Chen et al.'s HPD estimator: the shortest of the intervals [x_(j), x_(j + m)] of the sorted values.
        m = num_instances * 9 // 10
        sorted_X = np.sort(X, axis=0)
        start = (sorted_X[m:] - sorted_X[: num_instances - m]).argmin(axis=0)  # (num_features,)
        features = np.arange(num_features)
        sigma = X.std(axis=0)
        lower = sorted_X[start, features] - 3.0 * sigma
        upper = sorted_X[start + m, features] + 3.0 * sigma

        rng = check_random_state(self.random_state)
        num_splits = 2**self.max_depth - 1
        self.split_features_ = rng.randint(num_features, size=(self.num_trees, num_splits)).astype(
            np.min_scalar_type(num_features - 1)
        )
        ratios = rng.uniform(size=(self.num_trees, num_splits))
        self.split_points_ = np.empty((self.num_trees, num_splits))
        log_volumes = np.zeros((self.num_trees, 2 * num_splits + 1))
        for k in range(self.max_depth):
            nodes = np.arange(2**k - 1, 2 ** (k + 1) - 1)
            q = self.split_features_[:, nodes]  # (num_trees, 2^k)
            low, high = lower[q], upper[q]
            # The range of q at a node is set by its nearest ancestor that also splits q.
            for depth in range(k):  # from the root down, so that the nearest ancestor wins
                ancestors = ((nodes + 1) >> (k - depth)) - 1
                goes_right = (((nodes + 1) >> (k - depth - 1)) & 1) == 1
                splits_q = self.split_features_[:, ancestors] == q
                high = np.where(splits_q & ~goes_right, self.split_points_[:, ancestors], high)
                low = np.where(splits_q & goes_right, self.split_points_[:, ancestors], low)
            r = ratios[:, nodes]
            self.split_points_[:, nodes] = low + r * (high - low)
            # Lemma 2.2: the left child keeps the fraction r of the volume, the right child 1 - r.
            log_volumes[:, 2 * nodes + 1] = log_volumes[:, nodes] + np.log(r)
            log_volumes[:, 2 * nodes + 2] = log_volumes[:, nodes] + np.log1p(-r)
        # The density of a node per instance in its profile: 1 / (N * V(node)), as in Equation 5.
        self.node_weights_ = np.exp(-log_volumes) / self.window_size

        self.reference_profile_ = np.zeros(
            log_volumes.shape, dtype=np.min_scalar_type(self.window_size)
        )
        self.latest_profile_ = np.zeros_like(self.reference_profile_)
        flat = self.reference_profile_.reshape(-1)
        for x in X:
            flat[self._path(x)] += 1

    def _path(self, x: np.ndarray) -> np.ndarray:
        """Returns the flat node indices of the path of `x` in each tree, of shape (max_depth + 1, num_trees)."""
        split_features, split_points = (
            self.split_features_.reshape(-1),
            self.split_points_.reshape(-1),
        )
        path = np.empty((self.max_depth + 1, self.num_trees), dtype=np.intp)
        node = np.zeros(self.num_trees, dtype=np.intp)
        path[0] = self._node_offsets
        for k in range(self.max_depth):
            split = node + self._split_offsets
            goes_right = x[split_features[split]] >= split_points[split]
            node = 2 * node + 1 + goes_right
            path[k + 1] = node + self._node_offsets

        return path

    def _score_path(self, path: np.ndarray) -> float:
        counts = self.reference_profile_.reshape(-1)[path]  # (max_depth + 1, num_trees)
        stops = counts <= self.node_size_limit
        stops[-1] = True
        depth = stops.argmax(axis=0)  # the termination node of each tree
        density = (
            counts[depth, self._trees] * self.node_weights_.reshape(-1)[path[depth, self._trees]]
        )

        return -float(density.mean())

    def _fit_path(self, path: np.ndarray, y: int | None) -> None:
        if y != 1:
            self.latest_profile_.reshape(-1)[path] += 1
        self.num_recorded_ += 1
        if self.num_recorded_ == self.window_size:
            self.reference_profile_, self.latest_profile_ = (
                self.latest_profile_,
                self.reference_profile_,
            )
            self.latest_profile_.fill(0)
            self.num_recorded_ = 0

    def _fit_initial(self, x: np.ndarray, y: int | None) -> None:
        if y == 1:  # the paper builds the trees from normal instances only
            return
        if self.initial_window_ is None:
            self.initial_window_ = np.empty((self.window_size, len(x)))
        self.initial_window_[self.num_initial_] = x
        self.num_initial_ += 1
        if self.num_initial_ == self.window_size:
            self._build(self.initial_window_)
            self.initial_window_ = None

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> RSForest:
        """Records the next instance into the latest node profile, or into the first window until the trees are built.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): The label of the instance, 1 for an anomaly. An anomaly does not enter the profiles. None counts the instance as normal (Default=None).

        Returns:
            object: The model itself.
        """
        x = np.asarray(X, dtype=np.float64)
        if self.num_initial_ < self.window_size:
            self._fit_initial(x, y)
        else:
            self._fit_path(self._path(x), y)

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the next instance by its negative density under the reference node profiles, without recording it.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The negative density of the instance, so higher scores are more anomalous. It is 0.0 until the first window is full.
        """
        if self.num_initial_ < self.window_size:
            return 0.0

        return self._score_path(self._path(np.asarray(X, dtype=np.float64)))

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Scores the next instance against the reference node profiles, and then records it.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score and fit.
            y (int): The label of the instance, 1 for an anomaly. An anomaly does not enter the profiles. None counts the instance as normal (Default=None).

        Returns:
            float: The negative density of the instance. It is 0.0 until the first window is full.
        """
        x = np.asarray(X, dtype=np.float64)
        if self.num_initial_ < self.window_size:
            self._fit_initial(x, y)
            return 0.0

        path = self._path(x)
        score = self._score_path(path)
        self._fit_path(path, y)

        return score
