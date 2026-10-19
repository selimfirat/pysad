import heapq
from collections import deque

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class _Node:
    # A leaf holds a point ``x`` and its bounding box (1, num_features). A branch holds the
    # cut dimension and value and the bounding box (2, num_features) of the points below it.
    __slots__ = ("x", "dim", "cut", "left", "right", "parent", "count", "bbox")

    def __init__(self, x=None, dim=None, cut=None, left=None, right=None, count=1):
        self.x = x
        self.dim = dim
        self.cut = cut
        self.left = left
        self.right = right
        self.parent = None
        self.count = count
        self.bbox = None if x is None else x.reshape(1, -1)


class _RCTree:
    # A robust random cut tree, adapted from RCTree of rrcf 0.4.4 (rrcf/rrcf.py,
    # https://github.com/kLabUM/rrcf). It keeps the insert, forget, duplicate and codisp logic
    # and the order of the random draws, without the leaf depths, which codisp does not need
    # when it walks up to the root. For licensing information, see the end of this file.

    def __init__(self):
        self.root = None
        self.leaves = {}

    def find_duplicate(self, x):
        node = self.root
        while node.x is None:
            node = node.left if x[node.dim] <= node.cut else node.right
        return node if (node.x == x).all() else None

    @staticmethod
    def _cut(x, bbox, rng):
        # The InsertPoint cut: a dimension drawn in proportion to its span and a uniform value in it.
        # The trees run this for every node on every path, so it makes as few NumPy calls as it can.
        high = np.maximum(bbox[-1], x)
        # A leaf has a box of one row, where rrcf 0.4.4 writes the maximum over the minimum, so the
        # span is 0 and the cut falls on the first feature. This keeps that behavior until #331.
        low = high if len(bbox) == 1 else np.minimum(bbox[0], x)
        span = high - low
        r = rng.uniform(0, span.sum())
        span_sum = span.cumsum()
        # The first dimension whose cumulative span reaches r. span.sum() can round above the last
        # cumulative span, and then no dimension reaches r, so the first one is taken, as argmax does.
        dim = int(span_sum.searchsorted(r))
        if dim == len(span_sum):
            dim = 0
        return dim, low[dim] + span_sum[dim] - r

    @staticmethod
    def _children_bbox(node, out=None):
        # The bounding box of the two children, written into ``out`` if it is given.
        left, right = node.left.bbox, node.right.bbox
        out = np.empty((2, left.shape[1])) if out is None else out
        np.minimum(left[0], right[0], out=out[0])
        np.maximum(left[-1], right[-1], out=out[1])
        return out

    @staticmethod
    def _add_count(node, inc):
        while node is not None:
            node.count += inc
            node = node.parent

    def insert(self, x, index, rng):
        if self.root is None:
            self.root = self.leaves[index] = _Node(x=x)
            return

        duplicate = self.find_duplicate(x)
        if duplicate is not None:
            self._add_count(duplicate, 1)
            self.leaves[index] = duplicate
            return

        # Walk down until a cut separates x from the points below the node.
        leaf = _Node(x=x)
        node, parent = self.root, None
        while True:
            dim, cut = self._cut(x, node.bbox, rng)
            if cut <= node.bbox[0, dim]:
                branch = _Node(dim=dim, cut=cut, left=leaf, right=node, count=1 + node.count)
                break
            if cut >= node.bbox[-1, dim]:
                branch = _Node(dim=dim, cut=cut, left=node, right=leaf, count=1 + node.count)
                break
            parent = node
            node = node.left if x[node.dim] <= node.cut else node.right

        node.parent = leaf.parent = branch
        branch.parent = parent
        if parent is None:
            self.root = branch
        elif parent.left is node:
            parent.left = branch
        else:
            parent.right = branch
        self._add_count(parent, 1)

        # Expand the bounding boxes above the new branch.
        bbox = branch.bbox = self._children_bbox(branch)
        node = parent
        while node is not None:
            if not ((bbox[0] < node.bbox[0]).any() or (bbox[1] > node.bbox[1]).any()):
                break
            np.minimum(node.bbox[0], bbox[0], out=node.bbox[0])
            np.maximum(node.bbox[1], bbox[1], out=node.bbox[1])
            node = node.parent

        self.leaves[index] = leaf

    def forget(self, index):
        leaf = self.leaves.pop(index)
        if leaf.count > 1:
            self._add_count(leaf, -1)
            return
        if leaf is self.root:
            self.root = None
            return

        parent = leaf.parent
        sibling = parent.right if leaf is parent.left else parent.left
        if parent is self.root:
            sibling.parent = None
            self.root = sibling
            return

        grandparent = parent.parent
        sibling.parent = grandparent
        if parent is grandparent.left:
            grandparent.left = sibling
        else:
            grandparent.right = sibling
        self._add_count(grandparent, -1)

        # Shrink the bounding boxes that the forgotten point was on.
        node = grandparent
        while node is not None:
            if not ((node.bbox[0] == leaf.x) | (node.bbox[1] == leaf.x)).any():
                break
            self._children_bbox(node, out=node.bbox)
            node = node.parent

    def score(self, x, rng):
        # The collusive displacement of x if it were inserted: the largest ratio of sibling size
        # to subtree size on the path from the leaf of x to the root, without modifying the tree.
        if self.root is None:
            return 0.0

        node = self.find_duplicate(x)
        if node is not None:
            result = 0.0
        else:
            node = self.root
            while True:
                dim, cut = self._cut(x, node.bbox, rng)
                if cut <= node.bbox[0, dim] or cut >= node.bbox[-1, dim]:
                    break
                node = node.left if x[node.dim] <= node.cut else node.right
            result = float(node.count)

        while node.parent is not None:
            parent = node.parent
            sibling = parent.right if node is parent.left else parent.left
            result = max(result, sibling.count / (node.count + 1))
            node = parent
        return result


class RobustRandomCutForest(BaseModel):
    """Scores each instance by its collusive displacement in a forest of robust random cut trees over shingled inputs :cite:`guha2016robust`.

    Each point of the trees is a shingle, the concatenation of the last ``shingle_size`` instances, as in the experiments of the paper. Each tree is built on its own random sample of the stream, as in Section 2 of the paper: every instance gets the key ``time_decay * t + G`` in each tree, where ``t`` is its position in the stream and ``G`` is a standard Gumbel draw, and each tree keeps the ``tree_size`` instances with the largest keys. This is the weighted reservoir sample of :cite:`efraimidis2006weighted` with weights ``exp(time_decay * t)``. ``time_decay=0`` keeps a uniform random sample of the whole stream, as in the paper's synthetic experiment (reservoir sampling), and a larger ``time_decay`` favors recent instances, as in its taxi experiment.

    The trees follow the RCTree class of the `rrcf library <https://github.com/kLabUM/rrcf>`_ :cite:`bartos_2019_rrcf`.

    Args:
        num_trees (int): The number of trees. The paper uses 100 (synthetic data) and 200 (taxi data) (Default=100).
        shingle_size (int): The number of consecutive instances in a shingle, so each point of the trees has ``shingle_size * num_features`` values. The paper shingles a univariate stream with 4 (synthetic data) and 48 (taxi data) values. Until ``shingle_size - 1`` instances are fitted, there is no shingle and the score is 0. 1 uses each instance as it is (Default=1).
        tree_size (int): The number of instances in the sample of each tree (Default=256).
        time_decay (float): How strongly the samples favor recent instances. An instance that came ``k`` instances later is ``exp(time_decay * k)`` times as likely to be kept. 0 keeps a uniform sample of the whole stream (Default=0.0).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the cuts and the samples of the trees, which all share it. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self, num_trees=100, shingle_size=1, tree_size=256, time_decay=0.0, random_state=None
    ):
        self.tree_size = tree_size
        self.shingle_size = shingle_size
        self.num_trees = num_trees
        self.time_decay = time_decay
        self.random_state = random_state

        # One generator shared by all trees, since trees seeded alike would cut alike.
        self.rng = check_random_state(random_state)
        self.forest = [_RCTree() for _ in range(self.num_trees)]

        # The (key, index) pairs of the instances in each tree, as a min-heap on the key.
        self.samples = [[] for _ in range(self.num_trees)]
        # The last shingle_size - 1 fitted instances, which start the shingle of the next instance.
        self.recent = deque(maxlen=shingle_size - 1)
        self.index = 0

    def __getstate__(self):
        # NumPy's global random state is not pickled, so that the restored model keeps drawing from it.
        state = self.__dict__.copy()
        if self.rng is np.random.mtrand._rand:
            state["rng"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.rng = check_random_state(self.rng)

    def _shingle(self, X):
        # The shingle that ends with X, or None while fewer than shingle_size - 1 instances are fitted.
        if len(self.recent) < self.shingle_size - 1:
            return None

        return np.concatenate([*self.recent, np.ravel(X)])

    def fit_partial(self, X, y=None):
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        point = self._shingle(X)
        self.recent.append(np.ravel(X).copy())
        if point is None:
            return self

        for tree, sample in zip(self.forest, self.samples, strict=True):
            key = (self.time_decay * self.index + self.rng.gumbel(), self.index)
            if len(sample) < self.tree_size:
                heapq.heappush(sample, key)
            elif key > sample[0]:
                tree.forget(heapq.heapreplace(sample, key)[1])
            else:
                continue

            tree.insert(point, self.index, self.rng)

        self.index += 1

        return self

    def score_partial(self, X):
        """Scores the shingle that ends with the next instance by the collusive displacement it would have in each tree if it were inserted there, averaged over the trees. The trees are not modified, so only the random cuts draw from the random number generator.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        point = self._shingle(X)
        if point is None:
            return 0.0

        score = 0.0
        for tree in self.forest:
            score += tree.score(point, self.rng)

        return score / self.num_trees


# _RCTree is adapted from rrcf 0.4.4, which is distributed under the following license:
#
# MIT License
#
# Copyright (c) 2018 kLabUM
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
