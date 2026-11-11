from __future__ import annotations

import heapq
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.transform.preprocessing import ShingleTransformer


class _Node:
    # A leaf holds a point ``x`` and its bounding box (1, num_features). A branch holds the
    # cut dimension and value and the bounding box (2, num_features) of the points below it.
    __slots__ = ("x", "dim", "cut", "left", "right", "parent", "count", "bbox")

    def __init__(
        self,
        x: np.ndarray | None = None,
        dim: int | None = None,
        cut: float | None = None,
        left: _Node | None = None,
        right: _Node | None = None,
        count: int = 1,
    ) -> None:
        self.x = x
        self.dim = dim
        self.cut = cut
        self.left = left
        self.right = right
        self.parent: _Node | None = None
        self.count = count
        self.bbox: np.ndarray | None = None if x is None else x.reshape(1, -1)


class _ArenaNode:
    """Lazy object view over a single node slot in ``_ForestArena``."""

    __slots__ = ("_arena", "_slot")

    def __init__(self, arena: _ForestArena, slot: int) -> None:
        self._arena = arena
        self._slot = slot

    @property
    def x(self) -> np.ndarray | None:
        if self._arena.dim[self._slot] >= 0:
            return None
        return self._arena.bbox_low[self._slot]

    @property
    def dim(self) -> int | None:
        d = int(self._arena.dim[self._slot])
        return None if d < 0 else d

    @property
    def cut(self) -> float | None:
        if self._arena.dim[self._slot] < 0:
            return None
        return float(self._arena.cut[self._slot])

    @property
    def count(self) -> int:
        return int(self._arena.count[self._slot])

    @property
    def bbox(self) -> np.ndarray:
        s = self._slot
        if self._arena.dim[s] < 0:
            return self._arena.bbox_low[s].reshape(1, -1)
        return np.stack([self._arena.bbox_low[s], self._arena.bbox_high[s]])

    @property
    def left(self) -> _ArenaNode | None:
        c = int(self._arena.left[self._slot])
        return None if c < 0 else self._arena.view(c)

    @property
    def right(self) -> _ArenaNode | None:
        c = int(self._arena.right[self._slot])
        return None if c < 0 else self._arena.view(c)

    @property
    def parent(self) -> _ArenaNode | None:
        p = int(self._arena.parent[self._slot])
        return None if p < 0 else self._arena.view(p)


class _LeavesView(Mapping[Any, _ArenaNode]):
    """Mapping view from sample index to lazy ``_ArenaNode`` for ``_RCTree.leaves``."""

    __slots__ = ("_arena", "_slots")

    def __init__(self, arena: _ForestArena, slots: dict[Any, int]) -> None:
        self._arena = arena
        self._slots = slots

    def __getitem__(self, key: Any) -> _ArenaNode:
        return self._arena.view(self._slots[key])

    def __iter__(self) -> Iterator[Any]:
        return iter(self._slots)

    def __len__(self) -> int:
        return len(self._slots)

    def __contains__(self, key: object) -> bool:
        return key in self._slots


class _ForestArena:
    """Structure-of-Arrays node arena shared across all trees of a forest."""

    def __init__(self, num_trees: int, initial_cap: int) -> None:
        cap = max(16, initial_cap)
        self.num_trees = num_trees
        self.cap = cap
        self.num_features = 0
        self.roots = np.full(num_trees, -1, dtype=np.int32)
        self.left = np.full(cap, -1, dtype=np.int32)
        self.right = np.full(cap, -1, dtype=np.int32)
        self.parent = np.full(cap, -1, dtype=np.int32)
        self.sibling = np.full(cap, -1, dtype=np.int32)
        self.dim = np.full(cap, -1, dtype=np.int32)
        self.cut = np.zeros(cap, dtype=np.float64)
        self.count = np.zeros(cap, dtype=np.int32)
        self.bbox_low = np.empty((cap, 0), dtype=np.float64)
        self.bbox_high = np.empty((cap, 0), dtype=np.float64)
        self.free_slots = np.arange(cap - 1, -1, -1, dtype=np.int32)
        self.free_top = cap
        self._iota = np.arange(max(num_trees, 16), dtype=np.intp)
        self._buf_lo = np.empty((max(num_trees, 1), 0), dtype=np.float64)
        self._buf_sp = np.empty((max(num_trees, 1), 0), dtype=np.float64)
        self._views: dict[int, _ArenaNode] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_views"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._views = {}

    def __deepcopy__(self, memo: dict[int, Any]) -> _ForestArena:
        cls = self.__class__
        dup = cls.__new__(cls)
        memo[id(self)] = dup
        dup.num_trees = self.num_trees
        dup.cap = self.cap
        dup.num_features = self.num_features
        dup.roots = self.roots.copy()
        dup.left = self.left.copy()
        dup.right = self.right.copy()
        dup.parent = self.parent.copy()
        dup.sibling = self.sibling.copy()
        dup.dim = self.dim.copy()
        dup.cut = self.cut.copy()
        dup.count = self.count.copy()
        dup.bbox_low = self.bbox_low.copy()
        dup.bbox_high = self.bbox_high.copy()
        dup.free_slots = self.free_slots.copy()
        dup.free_top = self.free_top
        dup._iota = self._iota.copy()
        dup._buf_lo = self._buf_lo.copy()
        dup._buf_sp = self._buf_sp.copy()
        dup._views = {}
        return dup

    def view(self, slot: int) -> _ArenaNode:
        node = self._views.get(slot)
        if node is None:
            node = _ArenaNode(self, slot)
            self._views[slot] = node
        return node

    def ensure_features(self, num_features: int) -> None:
        if self.num_features == 0:
            self.num_features = num_features
            self.bbox_low = np.empty((self.cap, num_features), dtype=np.float64)
            self.bbox_high = np.empty((self.cap, num_features), dtype=np.float64)
            rows = max(self.num_trees, 1)
            self._buf_lo = np.empty((rows, num_features), dtype=np.float64)
            self._buf_sp = np.empty((rows, num_features), dtype=np.float64)

    def _ensure_free(self, needed: int) -> None:
        if self.free_top >= needed:
            return
        old_cap = self.cap
        new_cap = old_cap
        while self.free_top + (new_cap - old_cap) < needed:
            new_cap *= 2
        added = new_cap - old_cap
        for attr in ("left", "right", "parent", "sibling", "dim"):
            old = getattr(self, attr)
            arr = np.full(new_cap, -1, dtype=np.int32)
            arr[:old_cap] = old
            setattr(self, attr, arr)
        old_cut = self.cut
        self.cut = np.zeros(new_cap, dtype=np.float64)
        self.cut[:old_cap] = old_cut
        old_cnt = self.count
        self.count = np.zeros(new_cap, dtype=np.int32)
        self.count[:old_cap] = old_cnt
        if self.num_features > 0:
            old_lo = self.bbox_low
            self.bbox_low = np.empty((new_cap, self.num_features), dtype=np.float64)
            self.bbox_low[:old_cap] = old_lo
            old_hi = self.bbox_high
            self.bbox_high = np.empty((new_cap, self.num_features), dtype=np.float64)
            self.bbox_high[:old_cap] = old_hi
        else:
            self.bbox_low = np.empty((new_cap, 0), dtype=np.float64)
            self.bbox_high = np.empty((new_cap, 0), dtype=np.float64)

        new_free = np.empty(new_cap, dtype=np.int32)
        new_free[:added] = np.arange(new_cap - 1, old_cap - 1, -1, dtype=np.int32)
        new_free[added : added + self.free_top] = self.free_slots[: self.free_top]
        self.free_slots = new_free
        self.free_top += added
        self.cap = new_cap

    def alloc_one(self) -> int:
        self._ensure_free(1)
        self.free_top -= 1
        return int(self.free_slots[self.free_top])

    def alloc_many(self, k: int) -> np.ndarray:
        self._ensure_free(k)
        top = self.free_top
        self.free_top = top - k
        return self.free_slots[top - k : top].copy()

    def free_one(self, slot: int) -> None:
        if self._views:
            self._views.pop(slot, None)
        self.free_slots[self.free_top] = slot
        self.free_top += 1

    def find_duplicate_one(self, point: np.ndarray, root: int) -> int:
        curr = root
        while self.dim[curr] >= 0:
            d = int(self.dim[curr])
            curr = int(self.left[curr]) if point[d] <= self.cut[curr] else int(self.right[curr])
        return curr if bool((self.bbox_low[curr] == point).all()) else -1

    def find_duplicate_batch(
        self, point: np.ndarray, roots_sub: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        m = len(roots_sub)
        if m > len(self._iota):
            self._iota = np.arange(m, dtype=np.intp)
        leaves = roots_sub.copy()
        rem = self._iota[:m]
        curr = leaves
        dim = self.dim
        cut = self.cut
        left = self.left
        right = self.right
        while len(curr) > 8:
            d_cur = dim[curr]
            keep = d_cur >= 0
            if not keep.all():
                done = ~keep
                leaves[rem[done]] = curr[done]
                rem = rem[keep]
                if len(rem) == 0:
                    break
                curr = curr[keep]
                d_cur = d_cur[keep]
            curr = np.where(point[d_cur] <= cut[curr], left[curr], right[curr])
        if len(rem) > 0:
            for k in range(len(rem)):
                s = int(curr[k])
                while (d := int(dim[s])) >= 0:
                    s = int(left[s]) if point[d] <= cut[s] else int(right[s])
                leaves[rem[k]] = s
        is_dup = (self.bbox_low[leaves] == point).all(axis=1)
        return leaves, is_dup

    def cut_one_slot(
        self, point: np.ndarray, slot: int, rng: np.random.RandomState
    ) -> tuple[int, float, bool]:
        low = self.bbox_low[slot]
        high = self.bbox_high[slot]
        lo = np.minimum(low, point)
        span = np.maximum(high, point) - lo
        r = float(rng.uniform(0.0, float(span.sum())))
        span_cs = span.cumsum()
        d = int(span_cs.searchsorted(r))
        if d == len(span_cs):
            d = 0
        c_val = float(lo[d] + span_cs[d] - r)
        return d, c_val, bool(c_val <= low[d] or c_val >= high[d])

    def cut_slots(
        self, point: np.ndarray, slots: np.ndarray, rng: np.random.RandomState
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(slots)
        d = point.shape[0]
        if n > len(self._iota):
            self._iota = np.arange(n, dtype=np.intp)
        if n > self._buf_lo.shape[0]:
            self._buf_lo = np.empty((n, d), dtype=np.float64)
            self._buf_sp = np.empty((n, d), dtype=np.float64)
        row_idx = self._iota[:n]
        lows = self.bbox_low[slots]
        highs = self.bbox_high[slots]
        L = self._buf_lo[:n]
        spans = self._buf_sp[:n]
        np.minimum(lows, point, out=L)
        np.maximum(highs, point, out=spans)
        spans -= L
        span_sums = spans.sum(axis=1)
        r = rng.uniform(0.0, span_sums)
        np.cumsum(spans, axis=1, out=spans)

        dims = (spans < r[:, None]).sum(axis=1)
        if (dims == d).any():
            dims[dims == d] = 0

        cuts = L[row_idx, dims] + spans[row_idx, dims] - r
        stopped = (cuts <= lows[row_idx, dims]) | (cuts >= highs[row_idx, dims])
        return dims, cuts, stopped

    def codisp_sum(self, slots: np.ndarray, init_counts: np.ndarray) -> float:
        m = len(slots)
        if m > len(self._iota):
            self._iota = np.arange(m, dtype=np.intp)
        best = init_counts.astype(np.float64)
        rem = self._iota[:m]
        curr = slots
        parent = self.parent
        sibling = self.sibling
        count = self.count
        while len(curr) > 8:
            p = parent[curr]
            keep = p >= 0
            if not keep.all():
                rem = rem[keep]
                if len(rem) == 0:
                    return float(best.sum())
                curr = curr[keep]
                p = p[keep]
            ratios = count[sibling[curr]] / (count[curr] + 1)
            best_rem = best[rem]
            best[rem] = np.maximum(best_rem, ratios)
            curr = p
        for k in range(len(rem)):
            idx_k = int(rem[k])
            b = float(best[idx_k])
            c = int(curr[k])
            while (p_k := int(parent[c])) >= 0:
                r_k = int(count[sibling[c]]) / (int(count[c]) + 1)
                if r_k > b:
                    b = r_k
                c = p_k
            best[idx_k] = b
        return float(best.sum())

    def link_branch(
        self,
        tree_id: int,
        branch: int,
        leaf: int,
        node: int,
        parent: int,
        d: int,
        c_val: float,
    ) -> None:
        old_sib = int(self.sibling[node])
        if c_val <= self.bbox_low[node, d]:
            self.left[branch] = leaf
            self.right[branch] = node
        else:
            self.left[branch] = node
            self.right[branch] = leaf

        self.dim[branch] = d
        self.cut[branch] = c_val
        self.count[branch] = 1 + int(self.count[node])
        self.parent[leaf] = branch
        self.parent[node] = branch
        self.sibling[leaf] = node
        self.sibling[node] = leaf
        self.parent[branch] = parent

        if parent < 0:
            self.sibling[branch] = -1
            self.roots[tree_id] = branch
        else:
            self.sibling[branch] = old_sib
            self.sibling[old_sib] = branch
            if self.left[parent] == node:
                self.left[parent] = branch
            else:
                self.right[parent] = branch

    def unlink_leaf(self, tree_id: int, leaf: int) -> int:
        """Unlinks ``leaf`` (with count == 1) and its parent branch, freeing both; returns grandparent slot or -1."""
        if leaf == self.roots[tree_id]:
            self.roots[tree_id] = -1
            self.free_one(leaf)
            return -1

        p = int(self.parent[leaf])
        sib = int(self.sibling[leaf])
        if p == self.roots[tree_id]:
            self.parent[sib] = -1
            self.sibling[sib] = -1
            self.roots[tree_id] = sib
            self.free_one(leaf)
            self.free_one(p)
            return -1

        gp = int(self.parent[p])
        psib = int(self.sibling[p])
        self.parent[sib] = gp
        self.sibling[sib] = psib
        self.sibling[psib] = sib
        if self.left[gp] == p:
            self.left[gp] = sib
        else:
            self.right[gp] = sib
        self.free_one(leaf)
        self.free_one(p)
        return gp


class _RCTree:
    # A robust random cut tree, adapted from RCTree of rrcf 0.4.4 (rrcf/rrcf.py,
    # https://github.com/kLabUM/rrcf). It keeps the insert, forget, duplicate and codisp logic
    # and the order of the random draws, without the leaf depths, which codisp does not need
    # when it walks up to the root. For licensing information, see the end of this file.

    def __init__(self, arena: _ForestArena | None = None, tree_id: int = 0) -> None:
        self._arena = _ForestArena(1, 16) if arena is None else arena
        self._tree_id = tree_id
        self._slot_leaves: dict[Any, int] = {}
        self.leaves = _LeavesView(self._arena, self._slot_leaves)

    @property
    def root(self) -> _ArenaNode | None:
        r = int(self._arena.roots[self._tree_id])
        return None if r < 0 else self._arena.view(r)

    def find_duplicate(self, x: np.ndarray) -> _ArenaNode | None:
        r = int(self._arena.roots[self._tree_id])
        if r < 0:
            return None
        dup = self._arena.find_duplicate_one(np.asarray(x, dtype=np.float64), r)
        return None if dup < 0 else self._arena.view(dup)

    @staticmethod
    def _cut(x: np.ndarray, bbox: np.ndarray, rng: np.random.RandomState) -> tuple[int, float]:
        # The InsertPoint cut: a dimension drawn in proportion to its span and a uniform value in it.
        # The trees run this for every node on every path, so it makes as few NumPy calls as it can.
        high = np.maximum(bbox[-1], x)
        low = np.minimum(bbox[0], x)
        span = high - low
        r = rng.uniform(0, span.sum())
        span_sum = span.cumsum()
        # The first dimension whose cumulative span reaches r. span.sum() can round above the last
        # cumulative span, and then no dimension reaches r, so the first one is taken, as argmax does.
        dim = int(span_sum.searchsorted(r))
        if dim == len(span_sum):
            dim = 0
        return dim, float(low[dim] + span_sum[dim] - r)

    @staticmethod
    def _cut_batch(
        x: np.ndarray, nodes: list[Any], rng: np.random.RandomState
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not nodes:
            return np.empty(0, dtype=int), np.empty(0, dtype=float), np.empty(0, dtype=bool)

        d = len(x)
        lows = np.empty((len(nodes), d), dtype=np.float64)
        highs = np.empty((len(nodes), d), dtype=np.float64)
        for i, node in enumerate(nodes):
            bbox = node.bbox
            assert bbox is not None
            lows[i] = bbox[0]
            highs[i] = bbox[-1]

        H = np.maximum(highs, x)
        L = np.minimum(lows, x)
        spans = H - L
        span_sums = spans.sum(axis=1)
        r = rng.uniform(0, span_sums)
        span_cumsum = spans.cumsum(axis=1)

        dims = (span_cumsum < r[:, None]).sum(axis=1)
        dims = np.where(dims == d, 0, dims)

        row_idx = np.arange(len(nodes))
        cuts = L[row_idx, dims] + span_cumsum[row_idx, dims] - r

        low_dims = lows[row_idx, dims]
        high_dims = highs[row_idx, dims]
        stopped = (cuts <= low_dims) | (cuts >= high_dims)

        return dims, cuts, stopped

    def insert(self, x: np.ndarray, index: Any, rng: np.random.RandomState) -> None:
        point = np.asarray(x, dtype=np.float64).reshape(-1)
        arena = self._arena
        arena.ensure_features(len(point))
        tid = self._tree_id
        r = int(arena.roots[tid])
        if r < 0:
            leaf = arena.alloc_one()
            arena.left[leaf] = -1
            arena.right[leaf] = -1
            arena.parent[leaf] = -1
            arena.sibling[leaf] = -1
            arena.dim[leaf] = -1
            arena.count[leaf] = 1
            arena.bbox_low[leaf] = point
            arena.bbox_high[leaf] = point
            arena.roots[tid] = leaf
            self._slot_leaves[index] = leaf
            return

        dup = arena.find_duplicate_one(point, r)
        if dup >= 0:
            curr = dup
            while curr >= 0:
                arena.count[curr] += 1
                curr = int(arena.parent[curr])
            self._slot_leaves[index] = dup
            return

        node = r
        parent = -1
        while True:
            d, c_val, stop = arena.cut_one_slot(point, node, rng)
            if stop:
                break
            parent = node
            nd = int(arena.dim[node])
            node = int(arena.left[node]) if point[nd] <= arena.cut[node] else int(arena.right[node])

        leaf = arena.alloc_one()
        branch = arena.alloc_one()
        arena.left[leaf] = -1
        arena.right[leaf] = -1
        arena.dim[leaf] = -1
        arena.count[leaf] = 1
        arena.bbox_low[leaf] = point
        arena.bbox_high[leaf] = point

        arena.link_branch(tid, branch, leaf, node, parent, d, c_val)
        np.minimum(arena.bbox_low[node], point, out=arena.bbox_low[branch])
        np.maximum(arena.bbox_high[node], point, out=arena.bbox_high[branch])

        curr = parent
        while curr >= 0:
            arena.count[curr] += 1
            np.minimum(arena.bbox_low[curr], point, out=arena.bbox_low[curr])
            np.maximum(arena.bbox_high[curr], point, out=arena.bbox_high[curr])
            curr = int(arena.parent[curr])

        self._slot_leaves[index] = leaf

    def forget(self, index: Any) -> None:
        arena = self._arena
        leaf = self._slot_leaves.pop(index)
        if arena.count[leaf] > 1:
            curr = leaf
            while curr >= 0:
                arena.count[curr] -= 1
                curr = int(arena.parent[curr])
            return

        leaf_x = arena.bbox_low[leaf].copy()
        gp = arena.unlink_leaf(self._tree_id, leaf)
        if gp < 0:
            return

        curr = gp
        while curr >= 0:
            arena.count[curr] -= 1
            curr = int(arena.parent[curr])

        curr = gp
        while curr >= 0:
            if not bool(
                ((arena.bbox_low[curr] == leaf_x) | (arena.bbox_high[curr] == leaf_x)).any()
            ):
                break
            lc = int(arena.left[curr])
            rc = int(arena.right[curr])
            np.minimum(arena.bbox_low[lc], arena.bbox_low[rc], out=arena.bbox_low[curr])
            np.maximum(arena.bbox_high[lc], arena.bbox_high[rc], out=arena.bbox_high[curr])
            curr = int(arena.parent[curr])

    def score(self, x: np.ndarray, rng: np.random.RandomState) -> float:
        arena = self._arena
        r = int(arena.roots[self._tree_id])
        if r < 0:
            return 0.0
        point = np.asarray(x, dtype=np.float64).reshape(-1)
        dup = arena.find_duplicate_one(point, r)
        if dup >= 0:
            return arena.codisp_sum(np.array([dup], dtype=np.int32), np.array([0], dtype=np.int32))

        curr = r
        while True:
            _, _, stop = arena.cut_one_slot(point, curr, rng)
            if stop:
                break
            nd = int(arena.dim[curr])
            curr = int(arena.left[curr]) if point[nd] <= arena.cut[curr] else int(arena.right[curr])

        return arena.codisp_sum(
            np.array([curr], dtype=np.int32), np.array([int(arena.count[curr])], dtype=np.int32)
        )


class RobustRandomCutForest(BaseModel):
    """Scores each instance by its collusive displacement in a forest of robust random cut trees over shingled inputs :cite:`guha2016robust`.

    Each point of the trees is a shingle, the concatenation of the last ``shingle_size`` instances, as in the experiments of the paper. :class:`~pysad.transform.preprocessing.ShingleTransformer` builds the shingles, but the model uses no copies at the start of the stream. Each tree is built on its own random sample of the stream, as in Section 2 of the paper: every instance gets the key ``time_decay * t + G`` in each tree, where ``t`` is its position in the stream and ``G`` is a standard Gumbel draw, and each tree keeps the ``tree_size`` instances with the largest keys. This is the weighted reservoir sample of :cite:`efraimidis2006weighted` with weights ``exp(time_decay * t)``. ``time_decay=0`` keeps a uniform random sample of the whole stream, as in the paper's synthetic experiment (reservoir sampling), and a larger ``time_decay`` favors recent instances, as in its taxi experiment.

    The trees follow the RCTree class of the `rrcf library <https://github.com/kLabUM/rrcf>`_ :cite:`bartos_2019_rrcf`.

    Args:
        num_trees (int): The number of trees. The paper uses 100 (synthetic data) and 200 (taxi data) (Default=100).
        shingle_size (int): The number of consecutive instances in a shingle, so each point of the trees has ``shingle_size * num_features`` values. The paper shingles a univariate stream with 4 (synthetic data) and 48 (taxi data) values. Until ``shingle_size - 1`` instances are fitted, there is no shingle and the score is 0. 1 uses each instance as it is (Default=1).
        tree_size (int): The number of instances in the sample of each tree (Default=256).
        time_decay (float): How strongly the samples favor recent instances. An instance that came ``k`` instances later is ``exp(time_decay * k)`` times as likely to be kept. 0 keeps a uniform sample of the whole stream (Default=0.0).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the cuts and the samples of the trees, which all share it. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self,
        num_trees: int = 100,
        shingle_size: int = 1,
        tree_size: int = 256,
        time_decay: float = 0.0,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.tree_size = tree_size
        self.shingle_size = shingle_size
        self.num_trees = num_trees
        self.time_decay = time_decay
        self.random_state = random_state

        # One generator shared by all trees, since trees seeded alike would cut alike.
        self.rng: np.random.RandomState = check_random_state(random_state)
        init_cap = max(16, min(num_trees * 8, num_trees * (2 * max(tree_size, 1) + 2)))
        self._arena = _ForestArena(num_trees, init_cap)
        self.forest: list[_RCTree] = [_RCTree(self._arena, t) for t in range(self.num_trees)]

        # The (key, index) pairs of the instances in each tree, as a min-heap on the key.
        self.samples: list[list[tuple[float, int]]] = [[] for _ in range(self.num_trees)]
        self._min_keys = np.full(self.num_trees, -np.inf, dtype=np.float64)
        self._all_heaps_full = False
        self._done_tids = np.empty(num_trees, dtype=np.int32)
        self._done_nodes = np.empty(num_trees, dtype=np.int32)
        self._done_parents = np.empty(num_trees, dtype=np.int32)
        self._done_dims = np.empty(num_trees, dtype=np.intp)
        self._done_cuts = np.empty(num_trees, dtype=np.float64)
        self._final_slots = np.empty(num_trees, dtype=np.int32)
        self._init_counts = np.empty(num_trees, dtype=np.int32)
        # The last shingle_size - 1 fitted instances, which start the shingle of the next instance.
        self.shingler: ShingleTransformer = ShingleTransformer(shingle_size)
        self.index: int = 0

    def __getstate__(self) -> dict[str, Any]:
        # NumPy's global random state is not pickled, so that the restored model keeps drawing from it.
        state = self.__dict__.copy()
        if self.rng is np.random.mtrand._rand:
            state["rng"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.rng = check_random_state(self.rng)

    def _shingle(self, X: np.ndarray) -> np.ndarray | None:
        # The shingle that ends with X, or None while fewer than shingle_size - 1 instances are fitted.
        if not self.shingler.full:
            return None

        return self.shingler.transform_partial(X)

    def _forget_trees(self, evicted: list[tuple[int, int]]) -> None:
        arena = self._arena
        dec_list: list[int] = []
        shrink_gps: list[int] = []
        shrink_leaves: list[int] = []

        for t, ev_idx in evicted:
            leaf = self.forest[t]._slot_leaves.pop(ev_idx)
            if arena.count[leaf] > 1:
                dec_list.append(leaf)
                continue
            gp = arena.unlink_leaf(t, leaf)
            if gp >= 0:
                dec_list.append(gp)
                shrink_gps.append(gp)
                shrink_leaves.append(leaf)

        if dec_list:
            c_walk = np.array(dec_list, dtype=np.int32)
            while len(c_walk) > 0:
                arena.count[c_walk] -= 1
                c_walk = arena.parent[c_walk]
                c_walk = c_walk[c_walk >= 0]

        if shrink_gps:
            b_walk = np.array(shrink_gps, dtype=np.int32)
            pts = arena.bbox_low[np.array(shrink_leaves, dtype=np.int32)]
            while len(b_walk) > 0:
                on_bound = ((arena.bbox_low[b_walk] == pts) | (arena.bbox_high[b_walk] == pts)).any(
                    axis=1
                )
                if not on_bound.any():
                    break
                b_walk = b_walk[on_bound]
                pts = pts[on_bound]
                lc = arena.left[b_walk]
                rc = arena.right[b_walk]
                arena.bbox_low[b_walk] = np.minimum(arena.bbox_low[lc], arena.bbox_low[rc])
                arena.bbox_high[b_walk] = np.maximum(arena.bbox_high[lc], arena.bbox_high[rc])
                nxt = arena.parent[b_walk]
                keep = nxt >= 0
                b_walk = nxt[keep]
                pts = pts[keep]

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> RobustRandomCutForest:
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        point = self._shingle(X)
        self.shingler.fit_partial(X)
        if point is None:
            return self

        arena = self._arena
        arena.ensure_features(len(point))
        idx = self.index
        keys = self.time_decay * idx + self.rng.gumbel(size=self.num_trees)

        if self._all_heaps_full:
            accepted_arr = np.flatnonzero(keys > self._min_keys)
            if len(accepted_arr) == 0:
                self.index = idx + 1
                return self
            insert_tids = accepted_arr.tolist()
            evicted: list[tuple[int, int]] = []
            samples = self.samples
            min_keys = self._min_keys
            for t in insert_tids:
                sample = samples[t]
                ev_idx = heapq.heapreplace(sample, (float(keys[t]), idx))[1]
                min_keys[t] = sample[0][0]
                evicted.append((t, ev_idx))
            self._forget_trees(evicted)
        else:
            insert_tids = []
            evicted = []
            all_full = True
            for t, sample in enumerate(self.samples):
                key = (float(keys[t]), idx)
                if len(sample) < self.tree_size:
                    heapq.heappush(sample, key)
                    if len(sample) == self.tree_size:
                        self._min_keys[t] = sample[0][0]
                    else:
                        all_full = False
                    insert_tids.append(t)
                elif key > sample[0]:
                    ev_idx = heapq.heapreplace(sample, key)[1]
                    self._min_keys[t] = sample[0][0]
                    evicted.append((t, ev_idx))
                    insert_tids.append(t)
            if all_full and self.tree_size > 0:
                self._all_heaps_full = True
            if evicted:
                self._forget_trees(evicted)
            if not insert_tids:
                self.index = idx + 1
                return self

        tids_arr = np.array(insert_tids, dtype=np.int32)
        roots_sub = arena.roots[tids_arr]
        empty_mask = roots_sub < 0
        if empty_mask.any():
            empty_tids = tids_arr[empty_mask]
            new_leaves = arena.alloc_many(len(empty_tids))
            arena.left[new_leaves] = -1
            arena.right[new_leaves] = -1
            arena.parent[new_leaves] = -1
            arena.sibling[new_leaves] = -1
            arena.dim[new_leaves] = -1
            arena.count[new_leaves] = 1
            arena.bbox_low[new_leaves] = point
            arena.bbox_high[new_leaves] = point
            arena.roots[empty_tids] = new_leaves
            for t, lf in zip(empty_tids.tolist(), new_leaves.tolist(), strict=True):
                self.forest[t]._slot_leaves[idx] = lf
            tids_arr = tids_arr[~empty_mask]
            roots_sub = roots_sub[~empty_mask]

        if len(tids_arr) == 0:
            self.index = idx + 1
            return self

        cand_leaves, is_dup = arena.find_duplicate_batch(point, roots_sub)
        if is_dup.any():
            dup_tids = tids_arr[is_dup]
            dup_slots = cand_leaves[is_dup]
            for t, lf in zip(dup_tids.tolist(), dup_slots.tolist(), strict=True):
                self.forest[t]._slot_leaves[idx] = lf
            c_walk = dup_slots.copy()
            while len(c_walk) > 0:
                arena.count[c_walk] += 1
                c_walk = arena.parent[c_walk]
                c_walk = c_walk[c_walk >= 0]
            tids_arr = tids_arr[~is_dup]
            roots_sub = roots_sub[~is_dup]

        m = len(tids_arr)
        if m == 0:
            self.index = idx + 1
            return self

        active_tids: np.ndarray = tids_arr
        active_nodes: np.ndarray = roots_sub
        active_parents: np.ndarray = np.full(m, -1, dtype=np.int32)

        done_tids = self._done_tids
        done_nodes = self._done_nodes
        done_parents = self._done_parents
        done_dims = self._done_dims
        done_cuts = self._done_cuts
        n_done = 0

        while len(active_nodes) > 1:
            dims, cuts, stopped = arena.cut_slots(point, active_nodes, self.rng)
            if stopped.any():
                st_t = active_tids[stopped]
                ns = len(st_t)
                end = n_done + ns
                done_tids[n_done:end] = st_t
                done_nodes[n_done:end] = active_nodes[stopped]
                done_parents[n_done:end] = active_parents[stopped]
                done_dims[n_done:end] = dims[stopped]
                done_cuts[n_done:end] = cuts[stopped]
                n_done = end

                cont = ~stopped
                active_tids = active_tids[cont]
                active_parents = active_nodes[cont]
                active_nodes = active_nodes[cont]
                if len(active_nodes) == 0:
                    break
            else:
                active_parents = active_nodes

            d_act = arena.dim[active_nodes]
            active_nodes = np.where(
                point[d_act] <= arena.cut[active_nodes],
                arena.left[active_nodes],
                arena.right[active_nodes],
            )

        if len(active_nodes) == 1:
            t_1 = int(active_tids[0])
            s = int(active_nodes[0])
            p_1 = int(active_parents[0])
            rng = self.rng
            while True:
                d_1, c_1, stop_1 = arena.cut_one_slot(point, s, rng)
                if stop_1:
                    break
                p_1 = s
                nd = int(arena.dim[s])
                s = int(arena.left[s]) if point[nd] <= arena.cut[s] else int(arena.right[s])
            done_tids[n_done] = t_1
            done_nodes[n_done] = s
            done_parents[n_done] = p_1
            done_dims[n_done] = d_1
            done_cuts[n_done] = c_1
            n_done += 1

        d_tids = done_tids[:m]
        d_nodes = done_nodes[:m]
        d_parents = done_parents[:m]
        d_dims = done_dims[:m]
        d_cuts = done_cuts[:m]

        alloc = arena.alloc_many(2 * m)
        leaf_slots = alloc[:m]
        branch_slots = alloc[m:]

        arena.left[leaf_slots] = -1
        arena.right[leaf_slots] = -1
        arena.dim[leaf_slots] = -1
        arena.count[leaf_slots] = 1
        arena.bbox_low[leaf_slots] = point
        arena.bbox_high[leaf_slots] = point

        arena.bbox_low[branch_slots] = np.minimum(arena.bbox_low[d_nodes], point)
        arena.bbox_high[branch_slots] = np.maximum(arena.bbox_high[d_nodes], point)

        old_sibs = arena.sibling[d_nodes]
        go_left = d_cuts <= arena.bbox_low[d_nodes, d_dims]
        arena.left[branch_slots] = np.where(go_left, leaf_slots, d_nodes)
        arena.right[branch_slots] = np.where(go_left, d_nodes, leaf_slots)
        arena.dim[branch_slots] = d_dims
        arena.cut[branch_slots] = d_cuts
        arena.count[branch_slots] = 1 + arena.count[d_nodes]
        arena.parent[leaf_slots] = branch_slots
        arena.parent[d_nodes] = branch_slots
        arena.sibling[leaf_slots] = d_nodes
        arena.sibling[d_nodes] = leaf_slots
        arena.parent[branch_slots] = d_parents

        is_root = d_parents < 0
        if is_root.any():
            b_rt = branch_slots[is_root]
            arena.sibling[b_rt] = -1
            arena.roots[d_tids[is_root]] = b_rt
        has_p = ~is_root
        if has_p.any():
            b_hp = branch_slots[has_p]
            p_hp = d_parents[has_p]
            n_hp = d_nodes[has_p]
            s_hp = old_sibs[has_p]
            arena.sibling[b_hp] = s_hp
            arena.sibling[s_hp] = b_hp
            was_left = arena.left[p_hp] == n_hp
            arena.left[p_hp[was_left]] = b_hp[was_left]
            arena.right[p_hp[~was_left]] = b_hp[~was_left]

        forest = self.forest
        for t, lf in zip(d_tids.tolist(), leaf_slots.tolist(), strict=True):
            forest[t]._slot_leaves[idx] = lf

        anc = d_parents[has_p]
        if len(anc) > 0:
            c_walk = anc
            while len(c_walk) > 0:
                arena.count[c_walk] += 1
                c_walk = arena.parent[c_walk]
                c_walk = c_walk[c_walk >= 0]

            b_walk = anc
            while len(b_walk) > 0:
                lo_b = arena.bbox_low[b_walk]
                hi_b = arena.bbox_high[b_walk]
                outside = ((point < lo_b) | (point > hi_b)).any(axis=1)
                if not outside.any():
                    break
                b_walk = b_walk[outside]
                arena.bbox_low[b_walk] = np.minimum(lo_b[outside], point)
                arena.bbox_high[b_walk] = np.maximum(hi_b[outside], point)
                b_walk = arena.parent[b_walk]
                b_walk = b_walk[b_walk >= 0]

        self.index = idx + 1
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the shingle that ends with the next instance by the collusive displacement it would have in each tree if it were inserted there, averaged over the trees. The trees are not modified, so only the random cuts draw from the random number generator.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        point = self._shingle(X)
        if point is None:
            return 0.0

        arena = self._arena
        roots = arena.roots
        if len(roots) == 0:
            return 0.0
        non_empty = roots if roots.min() >= 0 else roots[roots >= 0]
        m = len(non_empty)
        if m == 0:
            return 0.0

        cand_leaves, is_dup = arena.find_duplicate_batch(point, non_empty)
        final_slots = self._final_slots[:m]
        init_counts = self._init_counts[:m]
        n_done = 0

        if is_dup.any():
            dup_slots = cand_leaves[is_dup]
            nd = len(dup_slots)
            final_slots[:nd] = dup_slots
            init_counts[:nd] = 0
            n_done = nd
            active = non_empty[~is_dup]
        else:
            active = non_empty

        while len(active) > 1:
            _, _, stopped = arena.cut_slots(point, active, self.rng)
            if stopped.any():
                st_slots = active[stopped]
                ns = len(st_slots)
                final_slots[n_done : n_done + ns] = st_slots
                init_counts[n_done : n_done + ns] = arena.count[st_slots]
                n_done += ns
                active = active[~stopped]
                if len(active) == 0:
                    break
            d_act = arena.dim[active]
            active = np.where(
                point[d_act] <= arena.cut[active],
                arena.left[active],
                arena.right[active],
            )

        if len(active) == 1:
            s = int(active[0])
            rng = self.rng
            while True:
                _, _, stop_1 = arena.cut_one_slot(point, s, rng)
                if stop_1:
                    break
                nd = int(arena.dim[s])
                s = int(arena.left[s]) if point[nd] <= arena.cut[s] else int(arena.right[s])
            final_slots[n_done] = s
            init_counts[n_done] = arena.count[s]

        return arena.codisp_sum(final_slots, init_counts) / self.num_trees


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
