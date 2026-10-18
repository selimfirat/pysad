import copy
import heapq
import sys
import types
from collections import deque
from importlib.metadata import version

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


def _import_rrcf():
    """Imports rrcf without pkg_resources, which setuptools>=82 no longer ships.

    rrcf only calls ``pkg_resources.get_distribution("rrcf").version``, so a stub serving that call is
    registered for the duration of the import and removed afterwards.
    """
    if "pkg_resources" in sys.modules:
        from rrcf import rrcf

        return rrcf

    stub = types.ModuleType("pkg_resources")
    stub.get_distribution = lambda name: types.SimpleNamespace(version=version(name))
    sys.modules["pkg_resources"] = stub
    try:
        from rrcf import rrcf
    finally:
        if sys.modules.get("pkg_resources") is stub:
            del sys.modules["pkg_resources"]

    return rrcf


class RobustRandomCutForest(BaseModel):
    """Scores each instance by its collusive displacement in a forest of robust random cut trees over shingled inputs :cite:`guha2016robust`.

    Each point of the trees is a shingle, the concatenation of the last ``shingle_size`` instances, as in the experiments of the paper. Each tree is built on its own random sample of the stream, as in Section 2 of the paper: every instance gets the key ``time_decay * t + G`` in each tree, where ``t`` is its position in the stream and ``G`` is a standard Gumbel draw, and each tree keeps the ``tree_size`` instances with the largest keys. This is the weighted reservoir sample of :cite:`efraimidis2006weighted` with weights ``exp(time_decay * t)``. ``time_decay=0`` keeps a uniform random sample of the whole stream, as in the paper's synthetic experiment (reservoir sampling), and a larger ``time_decay`` favors recent instances, as in its taxi experiment.

    The implementation uses `rrcf library <https://github.com/kLabUM/rrcf>`_ :cite:`bartos_2019_rrcf`.

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
        rrcf = _import_rrcf()

        self.tree_size = tree_size
        self.shingle_size = shingle_size
        self.num_trees = num_trees
        self.time_decay = time_decay
        self.random_state = random_state

        # A seed is turned into one generator shared by all trees, since trees seeded alike would cut alike.
        # None is passed on as is, so that rrcf draws from the ``numpy.random`` module.
        rng = None if random_state is None else check_random_state(random_state)

        self.forest = []
        for _ in range(self.num_trees):
            tree = rrcf.RCTree(random_state=rng)
            self.forest.append(tree)

        # The (key, index) pairs of the instances in each tree, as a min-heap on the key.
        self.samples = [[] for _ in range(self.num_trees)]
        # The last shingle_size - 1 fitted instances, which start the shingle of the next instance.
        self.recent = deque(maxlen=shingle_size - 1)
        self.index = 0

    def __getstate__(self):
        # rrcf.RCTree keeps a reference to the ``numpy.random`` module as its random number generator,
        # which cannot be pickled. Trees are shallow-copied without that reference and it is restored on load.
        state = self.__dict__.copy()
        state["forest"] = []
        for tree in self.forest:
            if tree.rng is np.random:
                tree = copy.copy(tree)
                tree.rng = None
            state["forest"].append(tree)

        return state

    def __setstate__(self, state):
        for tree in state["forest"]:
            if tree.rng is None:
                tree.rng = np.random

        self.__dict__.update(state)

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
            key = (self.time_decay * self.index + tree.rng.gumbel(), self.index)
            if len(sample) < self.tree_size:
                heapq.heappush(sample, key)
            elif key > sample[0]:
                tree.forget_point(heapq.heapreplace(sample, key)[1])
            else:
                continue

            tree.insert_point(point, index=self.index)

        self.index += 1

        return self

    def score_partial(self, X):
        """Scores the shingle that ends with the next instance by its collusive displacement in each tree after it is inserted there, averaged over the trees. The shingle is removed from the trees again, so only the random cuts of the insertions draw from the random number generator.

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
            tree.insert_point(point, index="test_point")
            score += tree.codisp("test_point")
            tree.forget_point("test_point")

        return score / self.num_trees

    def fit_score_partial(self, X, y=None):
        """Scores the next instance with the trees built on the previous instances, and then fits it, as in the experiments of the paper (Section 5).

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score and fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the input instance, as in `score_partial`.
        """
        score = self.score_partial(X)
        self.fit_partial(X, y)

        return score
