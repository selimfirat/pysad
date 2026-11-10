from __future__ import annotations

import numpy as np

from pysad.core.base_model import BaseModel

_FANOUT = 3  # M = L = 3, the fanout that the ClusTree experiments recommend
_SPLITS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))  # the 2 + 2 groups of 4 entries
_MIN_VARIANCE = 1e-50  # the variance of an entry without spread, as in MOA's ClusKernel
_VARIANCE_RTOL = 1e-12  # below this fraction of SS / n, the variance is rounding noise of the CF


class _Node:
    """A ClusTree node: the cluster features (n, LS, SS) of up to `_FANOUT` + 1 entries and their subtrees."""

    def __init__(self, num_features: int, ts: int) -> None:
        self.n = np.zeros(_FANOUT + 1)
        self.ls = np.zeros((_FANOUT + 1, num_features))
        self.ss = np.zeros((_FANOUT + 1, num_features))
        self.children: list[_Node] | None = None  # None for a leaf
        self.size = 0
        self.ts = ts  # the time of the last decay of all entries

    def means(self) -> np.ndarray:
        n = np.maximum(self.n[: self.size], np.finfo(np.float64).tiny)
        return self.ls[: self.size] / n[:, None]  # (size, d)

    def closest(self, x: np.ndarray) -> int:
        diff = self.means() - x
        return int(np.einsum("ij,ij->i", diff, diff).argmin())

    def append(self, n: float, ls: np.ndarray, ss: np.ndarray, child: _Node | None) -> None:
        self.n[self.size], self.ls[self.size], self.ss[self.size] = n, ls, ss
        if child is not None:
            assert self.children is not None
            self.children.append(child)
        self.size += 1

    def remove(self, j: int) -> None:
        last = self.size - 1
        self.n[j], self.ls[j], self.ss[j] = self.n[last], self.ls[last], self.ss[last]
        if self.children is not None:
            self.children[j] = self.children[last]
            self.children.pop()
        self.size = last

    def totals(self) -> tuple[float, np.ndarray, np.ndarray]:
        return (
            float(self.n[: self.size].sum()),
            self.ls[: self.size].sum(0),
            self.ss[: self.size].sum(0),
        )


class AnyOut(BaseModel):
    """Anytime outlier detection on a ClusTree (AnyOut) :cite:`assent2012anyout`.

    AnyOut keeps the stream in a ClusTree :cite:`kranen2011clustree`, a balanced tree of cluster features. Each entry of a node holds the decayed count n, linear sum LS and squared sum SS of the instances in its subtree. An instance descends from the root into the entry with the closest mean and becomes a new entry of a leaf. A node with more than 3 entries splits into two nodes of 2 entries, with the grouping of the least sum of distances between the means in each group. The weight of an instance decays as 2^(-`decay` dt) after dt more instances. AnyOut scores an instance against the entry where its descent stops when the time runs out: the root entries describe the stream coarsely, and each level below describes it in more detail. The mean outlier score (Definition 14.2 of :cite:`kranen2011anytime`) is the Euclidean distance from the instance to the mean of that entry. The density outlier score (Definition 14.3) is the Gaussian density at the instance with the mean and the variances of the entry.

    The DASFAA paper is not open access, so this class follows the description of AnyOut and the ClusTree in Chapters 11 and 14 of the thesis of the second author :cite:`kranen2011anytime`. The deviations:

    - The time budget is the number of tree levels `depth` that the descent of an instance reads, as in the anytime experiments of Section 14.4.3, where one time unit is one level. ``depth=None`` reads down to a leaf, which is the result with enough time. The class does not measure time, so the scores are reproducible.
    - The thesis scores an instance at the entry that it is inserted into. This class scores before it fits, as the thesis experiments score test instances on a tree built from the training instances, so the entry does not hold the scored instance.
    - The density g is higher for normal instances, so this class returns the negative log density, -log g. It ranks the instances in the same order as -g, and it does not round to 0 in many dimensions. A variance that is 0 or below the rounding error of the cluster feature, as in an entry of one instance, is raised to 1e-50, the floor of MOA's ClusKernel.
    - Each instance descends to a leaf when it is fitted, since a fit has no time limit. Thus the buffers and hitchhikers of interrupted insertions, and the aggregation of fast streams (Section 11.1.4), are not used.
    - A split that would make the tree taller than `max_height` levels is never made, as in MOA. The leaf then merges its closest pair of entries, the rule of the thesis for a node that has no time to split. Thus the tree holds at most 3^`max_height` leaf entries.
    - Before a node splits, it discards its entry with the smallest n if n < 2^(-`decay` `snapshot_interval`) (Eq. 11.1), and the cluster feature of the entry is subtracted from the entries above it. Since no snapshots are taken, `snapshot_interval` only sets this threshold.
    - The tree is built by insertion, not by the bulk loading that some experiments of the thesis use. The confidence 1 - e^(-s) of Section 14.4.4 only orders the instances of a constant stream for more computation time, so this class does not compute it.
    - The cluster features hold the sums of `x - ref`, where `ref` is the first fitted instance. Adding a constant to a feature then leaves every distance and variance unchanged, and the variance keeps its precision when the values are far from 0.

    Args:
        depth (int | None): The number of tree levels that the descent of an instance reads before it is scored, at least 1. None reads down to a leaf (Default=None).
        outlier_score (str): "mean" for the mean outlier score, which gives better rankings in the thesis experiments, or "density" for the density outlier score (Default="mean").
        max_height (int): The maximum number of tree levels, the default of MOA's ClusTree (Default=8).
        decay (float): The decay rate lambda per instance. An instance weighs half after 1 / `decay` more instances. 0 turns off the decay and the discarding of entries (Default=0.001).
        snapshot_interval (int): The snapshot interval t_snap of Eq. 11.1, in instances. A node about to split discards an entry that weighs less than 2^(-`decay` `snapshot_interval`) (Default=1000).
    """

    def __init__(
        self,
        depth: int | None = None,
        outlier_score: str = "mean",
        max_height: int = 8,
        decay: float = 0.001,
        snapshot_interval: int = 1000,
    ) -> None:
        if outlier_score not in ("mean", "density"):
            raise ValueError("outlier_score must be 'mean' or 'density'.")
        self.depth = depth
        self.outlier_score = outlier_score
        self.max_height = max_height
        self.decay = decay
        self.snapshot_interval = snapshot_interval
        self.root_: _Node | None = None
        self.height_ = 1
        self._time = 0
        # The cluster features hold sums relative to the first fitted instance.
        self._ref: np.ndarray | None = None

    def _decay_node(self, node: _Node) -> None:
        factor = 2.0 ** (-self.decay * (self._time - node.ts))
        node.n *= factor
        node.ls *= factor
        node.ss *= factor
        node.ts = self._time

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> AnyOut:
        """Inserts the next instance into the ClusTree.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        x = np.asarray(X, dtype=np.float64)
        if self.root_ is None:
            self._ref = x.copy()
            self.root_ = _Node(len(x), self._time)
        x = x - self._ref  # the cluster features hold the sums of x - ref

        path: list[tuple[_Node, int]] = []  # the inner nodes of the descent and their entries
        node = self.root_
        self._decay_node(node)
        while node.children is not None:
            i = node.closest(x)
            node.n[i] += 1.0
            node.ls[i] += x
            node.ss[i] += x * x
            path.append((node, i))
            node = node.children[i]
            self._decay_node(node)

        node.append(1.0, x, x * x, None)
        if node.size > _FANOUT:
            self._overflow(node, path)
        self._time += 1
        return self

    def _overflow(self, node: _Node, path: list[tuple[_Node, int]]) -> None:
        """Discards an outdated entry, splits up the path, or merges the closest leaf entries."""
        threshold = 2.0 ** (-self.decay * self.snapshot_interval)
        can_split = self.height_ < self.max_height or any(
            p.size < _FANOUT or p.n[: p.size].min() < threshold for p, _ in path
        )
        level = len(path)
        while node.size > _FANOUT:
            j = int(node.n.argmin())
            if node.n[j] < threshold:
                for parent, i in path[:level]:
                    parent.n[i] -= node.n[j]
                    parent.ls[i] -= node.ls[j]
                    parent.ss[i] -= node.ss[j]
                node.remove(j)
                return
            if not can_split:  # only a leaf, since every split up the path finds room
                self._merge_closest(node)
                return

            sibling = self._split(node)
            if level == 0:
                root = _Node(node.ls.shape[1], node.ts)
                root.children = []
                root.append(*node.totals(), node)
                root.append(*sibling.totals(), sibling)
                self.root_ = root
                self.height_ += 1
                return
            parent, i = path[level - 1]
            parent.n[i], parent.ls[i], parent.ss[i] = node.totals()
            parent.append(*sibling.totals(), sibling)
            node, level = parent, level - 1

    @staticmethod
    def _split(node: _Node) -> _Node:
        """Keeps one group of the 4 entries in `node` and moves the other group to a new node."""
        means = node.means()
        dists = np.linalg.norm(means[:, None] - means[None], axis=2)  # (4, 4)
        keep, move = min(_SPLITS, key=lambda s: dists[s[0]].item() + dists[s[1]].item())
        sibling = _Node(node.ls.shape[1], node.ts)
        if node.children is not None:
            sibling.children = []
        for j in move:
            sibling.append(
                node.n[j],
                node.ls[j],
                node.ss[j],
                None if node.children is None else node.children[j],
            )
        order = [*keep, *move]
        node.n[:] = node.n[order]
        node.ls[:] = node.ls[order]
        node.ss[:] = node.ss[order]
        if node.children is not None:
            node.children = [node.children[j] for j in keep]
        node.size = 2
        return sibling

    @staticmethod
    def _merge_closest(node: _Node) -> None:
        means = node.means()
        dists = np.linalg.norm(means[:, None] - means[None], axis=2)  # (4, 4)
        dists[np.tril_indices(node.size)] = np.inf
        a, b = np.unravel_index(dists.argmin(), dists.shape)
        node.n[a] += node.n[b]
        node.ls[a] += node.ls[b]
        node.ss[a] += node.ss[b]
        node.remove(int(b))

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the next instance against the closest entry at the level where its descent stops.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance. 0 before the first fit.
        """
        x = np.asarray(X, dtype=np.float64)
        if self.root_ is None:
            return 0.0
        x = x - self._ref

        node = self.root_
        i = node.closest(x)
        for _ in range(1, self.height_ if self.depth is None else self.depth):
            if node.children is None:
                break
            node = node.children[i]
            i = node.closest(x)

        n = max(node.n[i], np.finfo(np.float64).tiny)
        mean = node.ls[i] / n
        if self.outlier_score == "mean":
            return float(np.linalg.norm(x - mean))

        mean_sq = node.ss[i] / n
        variance = mean_sq - mean * mean
        variance = np.where(variance > _VARIANCE_RTOL * mean_sq, variance, _MIN_VARIANCE)
        return float(0.5 * ((x - mean) ** 2 / variance + np.log(2 * np.pi * variance)).sum())
