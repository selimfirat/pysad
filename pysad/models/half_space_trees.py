import copy
import numpy as np
from pysad.core.base_model import BaseModel


class HalfSpaceTrees(BaseModel):
    """Half-Space Trees method :cite:`tan2011fast`. Instances are scored against the reference mass profile built from the previous window before being recorded into the next one (Algorithm 3). Algorithm 3 does not score the first window; pysad scores each first-window instance that `initial_window_X` does not cover against the partial profile gathered so far in that window (the instances before it, without its own mass), so early instances look more anomalous, and the very first instance of a stream without `initial_window_X` scores 0.0.

    Args:
        feature_mins (np.float64 array of shape (num_features,)): Minimum boundary of the features.
        feature_maxes (np.float64 array of shape (num_features,)): Maximum boundary of the features.
        window_size (int): The size of the window (Default=100).
        num_trees (int): The number of treesint (Default=25).
        max_depth (int): Maximum depth of the trees (Default=15).
        initial_window_X (np.float64 array of shape (num_initial_instances,num_features)): The initial window to fit for initial calibration period. Per Tan et al. (IJCAI 2011), Algorithm 3, this is expected to hold the first `window_size` instances of the stream; they are fitted to build the reference mass profile and are not scored (Default=None).
    """

    def __init__(
            self,
            feature_mins,
            feature_maxes,
            window_size=100,
            num_trees=25,
            max_depth=15,
            initial_window_X=None):
        self.window_size = window_size
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.feature_maxes = feature_maxes
        self.feature_mins = feature_mins

        self.num_dimensions = len(self.feature_maxes)

        self.roots = [
            self._build_single_hs_tree(
                copy.deepcopy(
                    self.feature_mins),
                copy.deepcopy(
                    self.feature_maxes),
                0) for _ in range(
                self.num_trees)]

        self.is_first_window = True
        self.step = 0
        if initial_window_X is not None:
            self.fit(initial_window_X)

    def _build_single_hs_tree(self, mins, maxes, current_depth):
        if current_depth == self.max_depth:
            return self._Node(
                left=None,
                right=None,
                split_att=0,
                split_value=0.0,
                k=current_depth)

        q = np.random.randint(self.num_dimensions)
        p = (maxes[q] + mins[q]) / 2.0

        temp = maxes[q]
        maxes[q] = p
        left = self._build_single_hs_tree(
            copy.deepcopy(mins),
            copy.deepcopy(maxes),
            current_depth + 1)
        maxes[q] = temp
        mins[q] = p
        right = self._build_single_hs_tree(
            copy.deepcopy(mins),
            copy.deepcopy(maxes),
            current_depth + 1)

        return self._Node(
            left=left,
            right=right,
            split_att=q,
            split_value=p,
            k=current_depth)

    def _update_mass(self, x, node):
        node.l_mass += 1

        if node.k < self.max_depth:
            target_node = node.right if x[node.split_att] > node.split_value else node.left

            self._update_mass(x, target_node)

    def _update_model(self, node):

        if node is None:
            return

        self.is_first_window = False
        node.r_mass = node.l_mass
        node.l_mass = 0

        self._update_model(node.left)
        self._update_model(node.right)

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
            for root in self.roots:
                self._update_model(root)

        return self

    def _score_tree(self, X, node, first_window):
        if node is None:
            return 0.0

        target_node = node.right if X[node.split_att] > node.split_value else node.left
        # During the first window there is no reference yet, so score against the mass recorded so far.
        mass = node.l_mass if first_window else node.r_mass

        return mass * (2**node.k) + self._score_tree(X, target_node, first_window)

    def score_partial(self, X):
        """Scores the anomalousness of the next instance against the reference mass profile built from the previous window, without recording it.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance. During the first window there is no previous window to score against, and Algorithm 3 does not score these instances; pysad scores them against the partial profile gathered so far in that window instead, so early instances look more anomalous (0.0 when nothing has been recorded yet).
        """
        s = 0.0

        for root in self.roots:
            s += self._score_tree(X, root, self.is_first_window)

        return -s

    def fit_score_partial(self, X, y=None):
        """Scores the next instance against the reference mass profile built from the previous window (during the first window, against the partial profile gathered so far, as in `score_partial`), and then fits it, as Algorithm 3 does (score before updating the mass profile, and swap in the updated profile only at the end of a window).

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
        def __init__(self, left, right, split_att, split_value, k):
            self.left = left
            self.right = right
            self.r_mass = 0
            self.l_mass = 0
            self.split_att = split_att
            self.split_value = split_value
            self.k = k
