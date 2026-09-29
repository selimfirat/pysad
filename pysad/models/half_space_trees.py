import numbers

import numpy as np

from pysad.core.base_model import BaseModel


class HalfSpaceTrees(BaseModel):
    """Half-Space Trees method :cite:`tan2011fast`. Instances are scored against the reference mass profile built from the previous window before being recorded into the next one (Algorithm 3). Algorithm 3 does not score the first window: its first `window_size` instances only build the initial reference profile. pysad does score them, which departs from the paper (and from river and scikit-multiflow, which return a constant during the first window): each first-window instance that `initial_window_X` does not cover is scored against the partial profile of the n instances recorded before it in that window (without its own mass), and that score is rescaled by `window_size / n`, which estimates the mass a full window would hold, so that first-window scores are on the same scale as later ones. The very first instance of a stream without `initial_window_X` has n = 0 and scores 0.0. From the instance after the first window on, scoring follows Algorithm 3. As in the paper, each tree splits its own random work space by default: for every feature, a point s is drawn uniformly from [min, max] and the tree covers s ± 2 * max(s - min, max - s), so trees differ even on 1-D streams. The paper scores `r * 2^k` at the terminal node only (the node at maximum depth or the first one on the path holding at most sizeLimit instances), whereas pysad sums `r * 2^k` over every node on the path, with no sizeLimit early stop.

    Args:
        feature_mins (np.float64 array of shape (num_features,)): Minimum boundary of the features. Integer arrays are converted to float.
        feature_maxes (np.float64 array of shape (num_features,)): Maximum boundary of the features, at least `feature_mins`. Integer arrays are converted to float.
        window_size (int): The size of the window. Must be a positive integer (Default=100).
        num_trees (int): The number of trees. Must be a positive integer (Default=25).
        max_depth (int): Maximum depth of the trees. Must be a positive integer (Default=15).
        initial_window_X (np.float64 array of shape (num_initial_instances,num_features)): The initial window to fit for initial calibration period. Per Tan et al. (IJCAI 2011), Algorithm 3, this is expected to hold the first `window_size` instances of the stream; they are fitted to build the reference mass profile and are not scored (Default=None).
        random_work_space (bool): Whether each tree splits its own random work space, as in the paper. If False, every tree splits [feature_mins, feature_maxes], so on 1-D streams all trees are identical (Default=True).

    Raises:
        ValueError: If `window_size`, `num_trees` or `max_depth` is not a positive integer, if `random_work_space` is not a bool, or if `feature_mins` and `feature_maxes` are not finite 1-D arrays of the same length with every minimum at most its maximum.
    """

    # Index of the open window; a node's masses are brought up to it lazily (see `_roll_masses`).
    # The class-level defaults keep models pickled before the lazy swap loadable.
    current_window = 0

    def __init__(
        self,
        feature_mins,
        feature_maxes,
        window_size=100,
        num_trees=25,
        max_depth=15,
        initial_window_X=None,
        random_work_space=True,
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

        self.window_size = window_size
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.random_work_space = bool(random_work_space)
        self.feature_maxes = feature_maxes
        self.feature_mins = feature_mins

        self.num_dimensions = len(self.feature_maxes)

        self.roots = [
            self._build_single_hs_tree(*self._work_space(), 0) for _ in range(self.num_trees)
        ]

        self.is_first_window = True
        self.current_window = 0
        self.step = 0
        if initial_window_X is not None:
            self.fit(initial_window_X)

    def _work_space(self):
        if not self.random_work_space:
            return self.feature_mins.copy(), self.feature_maxes.copy()

        # Tan et al. (IJCAI 2011), Section 3.1: s ~ U(min, max), work range s ± 2 * max(s - min, max - s).
        s = np.random.uniform(self.feature_mins, self.feature_maxes)
        half_width = 2.0 * np.maximum(s - self.feature_mins, self.feature_maxes - s)

        return s - half_width, s + half_width

    def _build_single_hs_tree(self, mins, maxes, current_depth):
        if current_depth == self.max_depth:
            return self._Node(left=None, right=None, split_att=0, split_value=0.0, k=current_depth)

        q = np.random.randint(self.num_dimensions)
        p = (maxes[q] + mins[q]) / 2.0

        # Narrow the bounds in place for each subtree and restore them afterwards, instead of copying them.
        temp = maxes[q]
        maxes[q] = p
        left = self._build_single_hs_tree(mins, maxes, current_depth + 1)
        maxes[q] = temp
        temp = mins[q]
        mins[q] = p
        right = self._build_single_hs_tree(mins, maxes, current_depth + 1)
        mins[q] = temp

        return self._Node(left=left, right=right, split_att=q, split_value=p, k=current_depth)

    def _roll_masses(self, node):
        """Brings the masses of a node last visited before the open window up to it.

        Algorithm 3 sets r <- l and l <- 0 on every node when a window closes. Doing that eagerly makes the
        instance that closes a window visit all 2^(max_depth + 1) - 1 nodes of every tree, so it is deferred
        to the next visit of each node: l belongs to window `node.window`, and r to the window before it.
        """
        # A node not visited during the last closed window holds no mass from it.
        node.r_mass = node.l_mass if node.window == self.current_window - 1 else 0
        node.l_mass = 0
        node.window = self.current_window

    def _update_mass(self, x, node):
        if node.window != self.current_window:
            self._roll_masses(node)
        node.l_mass += 1

        if node.k < self.max_depth:
            target_node = node.right if x[node.split_att] > node.split_value else node.left

            self._update_mass(x, target_node)

    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: Returns the self.
        """
        self.step += 1

        for root in self.roots:
            self._update_mass(X, root)

        if self.step % self.window_size == 0:
            # Update model: r <- l on every node, applied lazily by `_roll_masses`.
            self.current_window += 1
            self.is_first_window = False

        return self

    def _score_tree(self, X, node, first_window):
        if node is None:
            return 0.0

        if node.window != self.current_window:
            self._roll_masses(node)
        target_node = node.right if X[node.split_att] > node.split_value else node.left
        # During the first window there is no reference yet, so score against the mass recorded so far.
        mass = node.l_mass if first_window else node.r_mass

        return mass * (2**node.k) + self._score_tree(X, target_node, first_window)

    def score_partial(self, X):
        """Scores the anomalousness of the next instance against the reference mass profile built from the previous window, without recording it.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance. During the first window there is no previous window to score against, and Algorithm 3 does not score these instances. pysad scores them against the partial profile of the n instances recorded so far in that window, rescaled by `window_size / n` to the scale of a full window (0.0 when nothing has been recorded yet). This rescaling is pysad's own and is not part of the paper.
        """
        s = 0.0

        for root in self.roots:
            s += self._score_tree(X, root, self.is_first_window)

        if self.is_first_window and self.step > 0:
            # Not in the paper: scale the partial profile of the `step` instances recorded so far
            # up to a full window, so first-window scores match the scale of later ones.
            s *= self.window_size / self.step

        return 0.0 - s

    def fit_score_partial(self, X, y=None):
        """Scores the next instance against the reference mass profile built from the previous window (during the first window, against the partial profile gathered so far, rescaled to a full window, as in `score_partial`), and then fits it, as Algorithm 3 does (score before updating the mass profile, and swap in the updated profile only at the end of a window).

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance, as in `score_partial`.
        """
        score = self.score_partial(X)
        self.fit_partial(X, y)

        return score

    class _Node:
        window = 0

        def __init__(self, left, right, split_att, split_value, k):
            self.left = left
            self.right = right
            self.r_mass = 0
            self.l_mass = 0
            self.split_att = split_att
            self.split_value = split_value
            self.k = k
            self.window = 0
