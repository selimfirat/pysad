import copy
import numpy as np
from pysad.core.base_model import BaseModel


class HalfSpaceTrees(BaseModel):
    """Half-Space Trees method :cite:`tan2011fast`.

    Args:
        feature_mins (np.float64 array of shape (num_features,)): Minimum boundary of the features.
        feature_maxes (np.float64 array of shape (num_features,)): Maximum boundary of the features.
        window_size (int): The size of the window (Default=100).
        num_trees (int): The number of treesint (Default=25).
        max_depth (int): Maximum depth of the trees (Default=15).
        initial_window_X (np.float64 array of shape (num_initial_instances,num_features)): The initial window to fit for initial calibration period. If not `None`, we simply apply fit to these instances (Default=None).
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
        if initial_window_X:
            self.fit(initial_window_X)

    def _build_single_hs_tree(self, mins, maxes, current_depth):
        if current_depth == self.max_depth:
            return self._Node(
                left=None,
                right=None,
                r_mass=0,
                l_mass=0,
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
            r_mass=0,
            l_mass=0,
            split_att=q,
            split_value=p,
            k=current_depth)

    def _update_mass(self, x, node, ref_window):
        if ref_window:
            node.r_mass += 1
            node.l_mass += 1  # Does not exist in original since we want it to predict while building the first window
        else:
            node.l_mass += 1

        if node.k < self.max_depth:
            target_node = node.right if x[node.split_att] > node.split_value else node.left

            self._update_mass(x, target_node, ref_window)

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
            self._update_mass(X, root, self.is_first_window)

        if self.step % self.window_size == 0:
            for root in self.roots:
                self._update_model(root)

        return self

    def _score_tree(self, X, node):
        if node is None:
            return 0.0

        target_node = node.right if X[node.split_att] > node.split_value else node.left

        return node.r_mass * (2**node.k) + self._score_tree(X, target_node)

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        s = 0.0

        for root in self.roots:
            s += self._score_tree(X, root)

        return -s

    class _Node:
        def __init__(self, left, right, r_mass, l_mass, split_att, split_value, k):
            self.left = left
            self.right = right
            self.r_mass = r_mass
            self.l_mass = l_mass
            self.split_att = split_att
            self.split_value = split_value
            self.k = k
