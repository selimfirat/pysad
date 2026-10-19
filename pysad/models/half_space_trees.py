import numbers

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class HalfSpaceTrees(BaseModel):
    """Scores each instance by how few instances of the previous window fall into the same nodes of random half-space trees :cite:`tan2011fast`.

    Instances are scored against the reference mass profile built from the previous window before being recorded into the next one (Algorithm 3). Algorithm 3 does not score the first window: its first `window_size` instances only build the initial reference profile. pysad does score them, which departs from the paper (and from river and scikit-multiflow, which return a constant during the first window): each first-window instance that `initial_window_X` does not cover is scored against the partial profile of the n instances recorded before it in that window (without its own mass), and that score is rescaled by `window_size / n`, which estimates the mass a full window would hold, so that first-window scores are on the same scale as later ones. The very first instance of a stream without `initial_window_X` has n = 0 and scores 0.0. From the instance after the first window on, scoring follows Algorithm 3. As in the paper, each tree splits its own random work space by default: for every feature, a point s is drawn uniformly from [min, max] and the tree covers s ± 2 * max(s - min, max - s), so trees differ even on 1-D streams. The paper scores `r * 2^k` at the terminal node only (the node at maximum depth or the first one on the path holding at most sizeLimit instances), whereas pysad sums `r * 2^k` over every node on the path, with no sizeLimit early stop. The sum uses the mass of every level, and on 17 ODDS datasets it gave a higher AUROC than the terminal-node score on 16 of them. Each tree is a full binary tree of depth `max_depth`, kept in NumPy arrays: a split feature and a float64 split value for each of its 2^max_depth - 1 internal nodes, and two masses for each of its 2^(max_depth + 1) - 1 nodes, in the smallest unsigned integer type that holds `window_size`. With the defaults, the 25 trees take about 11 MB.

    Args:
        feature_mins (np.float64 array of shape (num_features,)): Minimum boundary of the features. Integer arrays are converted to float.
        feature_maxes (np.float64 array of shape (num_features,)): Maximum boundary of the features, at least `feature_mins`. Integer arrays are converted to float.
        window_size (int): The size of the window. Must be a positive integer (Default=250, as in the paper).
        num_trees (int): The number of trees. Must be a positive integer (Default=25).
        max_depth (int): Maximum depth of the trees. Must be a positive integer (Default=15).
        initial_window_X (np.float64 array of shape (num_initial_instances,num_features)): The initial window to fit for initial calibration period. As in Algorithm 3 of :cite:`tan2011fast`, it holds the first `window_size` instances of the stream. The model fits them to build the reference mass profile and does not score them (Default=None).
        random_work_space (bool): Whether each tree splits its own random work space, as in the paper. If False, every tree splits [feature_mins, feature_maxes], so on 1-D streams all trees are identical (Default=True).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the work spaces and split features of the trees. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).

    Raises:
        ValueError: If `window_size`, `num_trees` or `max_depth` is not a positive integer, if `random_work_space` is not a bool, or if `feature_mins` and `feature_maxes` are not finite 1-D arrays of the same length with every minimum at most its maximum.
    """

    def __init__(
        self,
        feature_mins,
        feature_maxes,
        window_size=250,
        num_trees=25,
        max_depth=15,
        initial_window_X=None,
        random_work_space=True,
        random_state=None,
    ):
        for name, value in [
            ("window_size", window_size),
            ("num_trees", num_trees),
            ("max_depth", max_depth),
        ]:
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")

        if not isinstance(random_work_space, (bool, np.bool_)):
            raise ValueError(f"random_work_space must be a bool, got {random_work_space!r}.")

        # Float copies, so that integer bounds do not truncate the split values.
        feature_mins = np.array(feature_mins, dtype=np.float64)
        feature_maxes = np.array(feature_maxes, dtype=np.float64)
        if (
            feature_mins.ndim != 1
            or feature_mins.shape != feature_maxes.shape
            or feature_mins.size == 0
        ):
            raise ValueError(
                "feature_mins and feature_maxes must be non-empty 1-D arrays of the same length."
            )
        if not (np.all(np.isfinite(feature_mins)) and np.all(np.isfinite(feature_maxes))):
            raise ValueError("feature_mins and feature_maxes must be finite.")
        if np.any(feature_mins > feature_maxes):
            raise ValueError("feature_mins must not exceed feature_maxes.")

        self.window_size = int(window_size)
        self.max_depth = int(max_depth)
        self.num_trees = int(num_trees)
        self.random_work_space = bool(random_work_space)
        self.random_state = random_state
        self.feature_maxes = feature_maxes
        self.feature_mins = feature_mins

        self.num_dimensions = len(self.feature_maxes)

        # Full binary trees in heap order: node i has the children 2i + 1 and 2i + 2, so a node needs
        # no pointers, and only the 2^max_depth - 1 internal nodes hold a split. The feature indices and
        # the masses take the smallest unsigned integers that hold them: a node records at most
        # `window_size` instances per window.
        num_splits = 2**self.max_depth - 1
        self.split_atts = np.empty(
            (self.num_trees, num_splits), dtype=np.min_scalar_type(self.num_dimensions - 1)
        )
        self.split_values = np.empty((self.num_trees, num_splits))
        rng = check_random_state(random_state)
        for tree in range(self.num_trees):
            self.split_atts[tree], self.split_values[tree] = self._build_tree(rng)
        self.l_mass = np.zeros(
            (self.num_trees, 2 * num_splits + 1), dtype=np.min_scalar_type(self.window_size)
        )
        self.r_mass = np.zeros_like(self.l_mass)
        # r * 2^k of a node at depth k.
        self._depth_weights = 2 ** np.arange(self.max_depth + 1, dtype=np.int64)
        self._trees = np.arange(self.num_trees)

        self.is_first_window = True
        self.step = 0
        if initial_window_X is not None:
            self.fit(initial_window_X)

    def _work_space(self, rng):
        if not self.random_work_space:
            return self.feature_mins.copy(), self.feature_maxes.copy()

        # Tan et al. (IJCAI 2011), Section 3.1: s ~ U(min, max), work range s ± 2 * max(s - min, max - s).
        s = rng.uniform(self.feature_mins, self.feature_maxes)
        half_width = 2.0 * np.maximum(s - self.feature_mins, self.feature_maxes - s)

        return s - half_width, s + half_width

    def _build_tree(self, rng):
        """Draws the split features and computes the split values of one tree, in heap order.

        The tree halves its work space at every internal node: the node splits the feature drawn for it at the
        middle of the node's range. The features are drawn in depth-first order (a node, then its left subtree,
        then its right subtree), as a recursive build would draw them.
        """
        mins, maxes = self._work_space(rng)
        depth = self.max_depth
        atts_depth_first = rng.randint(self.num_dimensions, size=2**depth - 1)

        # Position in depth-first order of every internal node, level by level in heap order. The right child
        # of a node at depth k comes after the 2^(depth - k - 1) - 1 internal nodes of its left subtree.
        level = np.zeros(1, dtype=np.intp)
        levels = [level]
        for k in range(depth - 1):
            level = np.stack([level + 1, level + 2 ** (depth - k - 1)], axis=1).ravel()
            levels.append(level)
        atts = atts_depth_first[np.concatenate(levels)]

        # The range of feature q at a node is set by its nearest ancestor that also splits q (the upper bound
        # if the path goes left there, the lower bound if it goes right), or by the work space if none does.
        values = np.empty(2**depth - 1)
        for k in range(depth):
            nodes = np.arange(2**k - 1, 2 ** (k + 1) - 1)
            q = atts[nodes]
            lower, upper = mins[q], maxes[q]
            for ancestor_depth in range(k):  # from the root down, so that the nearest ancestor wins
                ancestors = ((nodes + 1) >> (k - ancestor_depth)) - 1
                goes_right = (((nodes + 1) >> (k - ancestor_depth - 1)) & 1) == 1
                splits_q = atts[ancestors] == q
                upper = np.where(splits_q & ~goes_right, values[ancestors], upper)
                lower = np.where(splits_q & goes_right, values[ancestors], lower)
            values[nodes] = (upper + lower) / 2.0

        return atts, values

    def _path(self, X):
        """Returns the index into the mass arrays of the max_depth + 1 nodes that `X` visits in each tree."""
        X = np.asarray(X)
        trees = self._trees
        path = np.zeros((self.num_trees, self.max_depth + 1), dtype=np.intp)
        for k in range(self.max_depth):
            node = path[:, k]
            goes_right = X[self.split_atts[trees, node]] > self.split_values[trees, node]
            path[:, k + 1] = 2 * node + 1 + goes_right

        return trees[:, None], path

    def _fit_path(self, path):
        self.step += 1
        self.l_mass[path] += 1

        if self.step % self.window_size == 0:
            # Update model: r <- l and l <- 0 on every node.
            self.r_mass, self.l_mass = self.l_mass, self.r_mass
            self.l_mass.fill(0)
            self.is_first_window = False

    def _score_path(self, path):
        # During the first window there is no reference yet, so score against the mass recorded so far.
        mass = (self.l_mass if self.is_first_window else self.r_mass)[path]
        s = float((mass * self._depth_weights).sum())

        if self.is_first_window and self.step > 0:
            # Not in the paper: scale the partial profile of the `step` instances recorded so far
            # up to a full window, so first-window scores match the scale of later ones.
            s *= self.window_size / self.step

        return 0.0 - s

    def fit_partial(self, X, y=None):
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        self._fit_path(self._path(X))

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance against the reference mass profile built from the previous window, without recording it.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance. During the first window there is no previous window to score against, and Algorithm 3 does not score these instances. pysad scores them against the partial profile of the n instances recorded so far in that window, rescaled by `window_size / n` to the scale of a full window (0.0 when nothing has been recorded yet). This rescaling is pysad's own and is not part of the paper.
        """
        return self._score_path(self._path(X))

    def fit_score_partial(self, X, y=None):
        """Scores the next instance against the reference mass profile, and then updates the model with it.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score and fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance.
        """
        path = self._path(X)
        score = self._score_path(path)
        self._fit_path(path)

        return score
