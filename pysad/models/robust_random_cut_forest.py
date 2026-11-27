from __future__ import annotations

import heapq
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from pysad import _kernels
from pysad.core.base_model import BaseModel
from pysad.explainability import treeshap_ensemble
from pysad.transform.preprocessing import ShingleTransformer


class _ForestArena:
    """Structure-of-Arrays node arena shared across all trees of a forest."""

    # Tree cut, duplicate, and collusive-displacement logic is adapted from RCTree of rrcf 0.4.4
    # (https://github.com/kLabUM/rrcf); see license notice at the bottom of this file.

    def __init__(self, num_trees: int, initial_cap: int) -> None:
        cap = max(16, initial_cap)
        self.num_trees = num_trees
        self.cap = cap
        self.num_features = 0
        self.roots: NDArray[np.int32] = np.full(num_trees, -1, dtype=np.int32)
        self.left: NDArray[np.int32] = np.full(cap, -1, dtype=np.int32)
        self.right: NDArray[np.int32] = np.full(cap, -1, dtype=np.int32)
        self.parent: NDArray[np.int32] = np.full(cap, -1, dtype=np.int32)
        self.sibling: NDArray[np.int32] = np.full(cap, -1, dtype=np.int32)
        self.dim: NDArray[np.int32] = np.full(cap, -1, dtype=np.int32)
        self.cut: NDArray[np.float64] = np.zeros(cap, dtype=np.float64)
        self.count: NDArray[np.int32] = np.zeros(cap, dtype=np.int32)
        self.bbox_low: NDArray[np.float64] = np.empty((cap, 0), dtype=np.float64)
        self.bbox_high: NDArray[np.float64] = np.empty((cap, 0), dtype=np.float64)
        self.free_slots: NDArray[np.int32] = np.arange(cap - 1, -1, -1, dtype=np.int32)
        self.free_top = cap
        self._iota: NDArray[np.intp] = np.arange(max(num_trees, 16), dtype=np.intp)
        self._buf_lo: NDArray[np.float64] = np.empty((max(num_trees, 1), 0), dtype=np.float64)
        self._buf_sp: NDArray[np.float64] = np.empty((max(num_trees, 1), 0), dtype=np.float64)

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

    def alloc_many(self, k: int) -> NDArray[np.int32]:
        self._ensure_free(k)
        top = self.free_top
        self.free_top = top - k
        return self.free_slots[top - k : top].copy()

    def free_one(self, slot: int) -> None:
        self.free_slots[self.free_top] = slot
        self.free_top += 1

    def find_duplicate_batch(
        self, point: NDArray[np.float64], roots_sub: NDArray[np.int32]
    ) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
        rust_route = _kernels.rrcf_route_to_leaf
        if rust_route is not None and point.ndim == 1 and point.shape[0] >= self.num_features:
            leaves = rust_route(
                np.asarray(point, dtype=np.float64),
                roots_sub.astype(np.int32, copy=False),
                self.dim,
                self.cut,
                self.left,
                self.right,
            )
            is_dup = (self.bbox_low[leaves] == point).all(axis=1)
            return leaves, is_dup

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
                curr = curr[keep]
                d_cur = d_cur[keep]
            curr = np.where(point[d_cur] <= cut[curr], left[curr], right[curr])
        for k in range(len(rem)):
            s = int(curr[k])
            while (d := int(dim[s])) >= 0:
                s = int(left[s]) if point[d] <= cut[s] else int(right[s])
            leaves[rem[k]] = s
        is_dup = (self.bbox_low[leaves] == point).all(axis=1)
        return leaves, is_dup

    def cut_one_slot(
        self, point: NDArray[np.float64], slot: int, rng: np.random.RandomState
    ) -> tuple[int, float, bool]:
        low = self.bbox_low[slot]
        high = self.bbox_high[slot]
        lo = np.minimum(low, point)
        span = np.maximum(high, point) - lo
        r = float(rng.uniform(0.0, float(span.sum())))
        span_cs = span.cumsum()
        # If span.sum() rounds above span_cs[-1], fall back to dimension 0 as argmax does.
        d = int(span_cs.searchsorted(r))
        if d == len(span_cs):
            d = 0
        c_val = float(lo[d] + span_cs[d] - r)
        return d, c_val, bool(c_val <= low[d] or c_val >= high[d])

    def find_cut(
        self, point: NDArray[np.float64], slot: int, parent: int, rng: np.random.RandomState
    ) -> tuple[int, int, int, float]:
        """Return the first separating cut and its node and parent."""
        while True:
            dim, cut, stopped = self.cut_one_slot(point, slot, rng)
            if stopped:
                return slot, parent, dim, cut
            parent = slot
            dim = int(self.dim[slot])
            slot = int(self.left[slot]) if point[dim] <= self.cut[slot] else int(self.right[slot])

    def cut_slots(
        self, point: NDArray[np.float64], slots: NDArray[np.int32], rng: np.random.RandomState
    ) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.bool_]]:
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

    def codisp_sum(self, slots: NDArray[np.int32], init_counts: NDArray[np.int32]) -> float:
        rust_codisp = _kernels.rrcf_codisp_sum
        if rust_codisp is not None:
            return float(
                rust_codisp(
                    slots.astype(np.int32, copy=False),
                    init_counts.astype(np.int32, copy=False),
                    self.parent,
                    self.sibling,
                    self.count,
                )
            )

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
        self._slot_leaves: list[dict[int, int]] = [{} for _ in range(self.num_trees)]

        # The (key, index) pairs of the instances in each tree, as a min-heap on the key.
        self.samples: list[list[tuple[float, int]]] = [[] for _ in range(self.num_trees)]
        self._min_keys: NDArray[np.float64] = np.full(self.num_trees, -np.inf, dtype=np.float64)
        self._all_heaps_full = False
        self._done_tids: NDArray[np.int32] = np.empty(num_trees, dtype=np.int32)
        self._done_nodes: NDArray[np.int32] = np.empty(num_trees, dtype=np.int32)
        self._done_parents: NDArray[np.int32] = np.empty(num_trees, dtype=np.int32)
        self._done_dims: NDArray[np.intp] = np.empty(num_trees, dtype=np.intp)
        self._done_cuts: NDArray[np.float64] = np.empty(num_trees, dtype=np.float64)
        self._final_slots: NDArray[np.int32] = np.empty(num_trees, dtype=np.int32)
        self._init_counts: NDArray[np.int32] = np.empty(num_trees, dtype=np.int32)
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

    def _shingle(self, X: NDArray[np.float64]) -> NDArray[np.float64] | None:
        # The shingle that ends with X, or None while fewer than shingle_size - 1 instances are fitted.
        if not self.shingler.full:
            return None

        return self.shingler.transform_partial(X)

    def _update_counts(self, slots: NDArray[np.int32], delta: int) -> None:
        """Update each node and its ancestors by the same count."""
        arena = self._arena
        while len(slots) > 0:
            arena.count[slots] += delta
            slots = arena.parent[slots]
            slots = slots[slots >= 0]

    def _forget_trees(self, evicted: list[tuple[int, int]]) -> None:
        arena = self._arena
        slot_leaves = self._slot_leaves
        dec_list: list[int] = []
        shrink_gps: list[int] = []
        shrink_leaves: list[int] = []

        for t, ev_idx in evicted:
            leaf = slot_leaves[t].pop(ev_idx)
            if arena.count[leaf] > 1:
                dec_list.append(leaf)
                continue
            gp = arena.unlink_leaf(t, leaf)
            if gp >= 0:
                dec_list.append(gp)
                shrink_gps.append(gp)
                shrink_leaves.append(leaf)

        if dec_list:
            self._update_counts(np.array(dec_list, dtype=np.int32), -1)

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

    def fit_partial(self, X: NDArray[np.float64], y: int | None = None) -> RobustRandomCutForest:
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

        self._arena.ensure_features(len(point))
        idx = self.index
        keys = self.time_decay * idx + self.rng.gumbel(size=self.num_trees)
        insert_tids = self._sample_trees(keys, idx)
        if insert_tids:
            self._insert_point(point, np.array(insert_tids, dtype=np.int32), idx)
        self.index = idx + 1
        return self

    def _sample_trees(self, keys: NDArray[np.float64], idx: int) -> list[int]:
        """Update reservoir samples and return the trees that accept the point."""
        if not self._all_heaps_full:
            return self._fill_samples(keys, idx)

        insert_tids: list[int] = np.flatnonzero(keys > self._min_keys).tolist()
        evicted: list[tuple[int, int]] = []
        for t in insert_tids:
            sample = self.samples[t]
            ev_idx = heapq.heapreplace(sample, (float(keys[t]), idx))[1]
            self._min_keys[t] = sample[0][0]
            evicted.append((t, ev_idx))
        if evicted:
            self._forget_trees(evicted)
        return insert_tids

    def _fill_samples(self, keys: NDArray[np.float64], idx: int) -> list[int]:
        """Fill reservoir samples until all trees reach their size limit."""
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
        return insert_tids

    def _insert_point(
        self, point: NDArray[np.float64], tids_arr: NDArray[np.int32], idx: int
    ) -> None:
        """Insert empty roots and duplicates before drawing new branches."""
        arena = self._arena
        slot_leaves = self._slot_leaves
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
                slot_leaves[t][idx] = lf
            tids_arr = tids_arr[~empty_mask]
            roots_sub = roots_sub[~empty_mask]

        if len(tids_arr) == 0:
            return

        cand_leaves, is_dup = arena.find_duplicate_batch(point, roots_sub)
        if is_dup.any():
            dup_tids = tids_arr[is_dup]
            dup_slots = cand_leaves[is_dup]
            for t, lf in zip(dup_tids.tolist(), dup_slots.tolist(), strict=True):
                slot_leaves[t][idx] = lf
            self._update_counts(dup_slots, 1)
            tids_arr = tids_arr[~is_dup]
            roots_sub = roots_sub[~is_dup]

        if len(tids_arr) > 0:
            self._insert_branches(point, tids_arr, roots_sub, idx)

    def _find_cuts(
        self, point: NDArray[np.float64], tids_arr: NDArray[np.int32], roots_sub: NDArray[np.int32]
    ) -> None:
        """Store insertion cuts in the forest buffers, in traversal order."""
        arena = self._arena
        m = len(tids_arr)
        active_tids: NDArray[np.int32] = tids_arr
        active_nodes: NDArray[np.int32] = roots_sub
        active_parents: NDArray[np.int32] = np.full(m, -1, dtype=np.int32)

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
            s, p_1, d_1, c_1 = arena.find_cut(point, s, p_1, self.rng)
            done_tids[n_done] = t_1
            done_nodes[n_done] = s
            done_parents[n_done] = p_1
            done_dims[n_done] = d_1
            done_cuts[n_done] = c_1
            n_done += 1

    def _insert_branches(
        self,
        point: NDArray[np.float64],
        tids_arr: NDArray[np.int32],
        roots_sub: NDArray[np.int32],
        idx: int,
    ) -> None:
        """Attach new leaves and branches, then update their ancestors."""
        self._find_cuts(point, tids_arr, roots_sub)
        arena = self._arena
        m = len(tids_arr)
        d_tids = self._done_tids[:m]
        d_nodes = self._done_nodes[:m]
        d_parents = self._done_parents[:m]
        d_dims = self._done_dims[:m]
        d_cuts = self._done_cuts[:m]

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

        for t, lf in zip(d_tids.tolist(), leaf_slots.tolist(), strict=True):
            self._slot_leaves[t][idx] = lf

        anc = d_parents[has_p]
        self._update_counts(anc, 1)

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

    def score_partial(self, X: NDArray[np.float64]) -> float:
        """Scores the shingle that ends with the next instance by the collusive displacement it would have in each tree if it were inserted there, averaged over the trees. The trees are not modified, and the random number generator state is restored afterwards so that scoring does not advance the model's RNG.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        rng_state = self.rng.get_state()
        try:
            point = self._shingle(X)
            return 0.0 if point is None else self._score_shingle(point)
        finally:
            self.rng.set_state(rng_state)

    def _score_shingle(self, point: NDArray[np.float64]) -> float:
        """Score a complete shingle without changing the trees."""
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

        self._score_slots(point, active, n_done, final_slots, init_counts)
        return arena.codisp_sum(final_slots, init_counts) / self.num_trees

    def _score_slots(
        self,
        point: NDArray[np.float64],
        active: NDArray[np.int32],
        n_done: int,
        final_slots: NDArray[np.int32],
        init_counts: NDArray[np.int32],
    ) -> None:
        """Store hypothetical insertion nodes and counts for distinct points."""
        arena = self._arena
        while len(active) > 1:
            _, _, stopped = arena.cut_slots(point, active, self.rng)
            if stopped.any():
                st_slots = active[stopped]
                ns = len(st_slots)
                final_slots[n_done : n_done + ns] = st_slots
                init_counts[n_done : n_done + ns] = arena.count[st_slots]
                n_done += ns
                active = active[~stopped]
            d_act = arena.dim[active]
            active = np.where(
                point[d_act] <= arena.cut[active],
                arena.left[active],
                arena.right[active],
            )

        if len(active) == 1:
            s = int(active[0])
            s, _, _, _ = arena.find_cut(point, s, -1, self.rng)
            final_slots[n_done] = s
            init_counts[n_done] = arena.count[s]

    def _collect_tree_structure(
        self, tree_id: int
    ) -> tuple[
        NDArray[np.intp],
        NDArray[np.float64],
        NDArray[np.intp],
        NDArray[np.intp],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Collect the structure of a single tree for TreeSHAP.

        Returns:
            Tuple of (features, thresholds, left, right, values, left_fractions) for the tree.
            Arrays are indexed by node ID (0 to num_nodes-1 in the collected tree).
        """
        arena = self._arena
        root = arena.roots[tree_id]
        if root < 0:
            # Empty tree
            return (
                np.array([-1], dtype=np.intp),
                np.array([0.0], dtype=np.float64),
                np.array([-1], dtype=np.intp),
                np.array([-1], dtype=np.intp),
                np.array([0.0], dtype=np.float64),
                np.array([0.5], dtype=np.float64),
            )

        # Traverse the tree to collect all nodes in BFS order
        nodes = []
        queue = [root]
        node_to_idx = {root: 0}
        next_idx = 1
        while queue:
            node = queue.pop(0)
            nodes.append(node)
            left_child = arena.left[node]
            right_child = arena.right[node]
            if left_child >= 0 and left_child not in node_to_idx:
                node_to_idx[left_child] = next_idx
                next_idx += 1
                queue.append(left_child)
            if right_child >= 0 and right_child not in node_to_idx:
                node_to_idx[right_child] = next_idx
                next_idx += 1
                queue.append(right_child)

        num_nodes = len(nodes)
        features = np.full(num_nodes, -1, dtype=np.intp)
        thresholds = np.zeros(num_nodes, dtype=np.float64)
        left = np.full(num_nodes, -1, dtype=np.intp)
        right = np.full(num_nodes, -1, dtype=np.intp)
        values = np.zeros(num_nodes, dtype=np.float64)
        left_fractions = np.full(num_nodes, 0.5, dtype=np.float64)

        for idx, node in enumerate(nodes):
            d = arena.dim[node]
            if d >= 0:  # Internal node
                features[idx] = d
                thresholds[idx] = arena.cut[node]
                left_child = arena.left[node]
                right_child = arena.right[node]
                left[idx] = node_to_idx[left_child]
                right[idx] = node_to_idx[right_child]
                total = arena.count[left_child] + arena.count[right_child]
                left_fractions[idx] = arena.count[left_child] / total
            else:
                values[idx] = arena.codisp_sum(
                    np.array([node], dtype=np.int32), np.zeros(1, dtype=np.int32)
                )

        return features, thresholds, left, right, values, left_fractions

    def score_explain_partial(self, X: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Returns the anomaly score and per-feature SHAP values for the next instance.

        Uses the TreeSHAP algorithm :cite:`lundberg2020consistent` to compute exact SHAP values
        for the forest of random cut trees. The SHAP values explain the collusive displacement
        (CoDisp) of the leaf each instance falls into, averaged across trees.
        Positive SHAP values indicate features that increase the anomaly score (more anomalous),
        negative values indicate features that decrease the anomaly score (more normal).

        Like ``score_partial``, it does not update the model or record the instance. Before
        the first shingle is fitted, the method returns 0.0 and zeros of shape
        ``(shingle_size * num_features,)``.

        Note: The explanation uses the current tree structure without the random cuts that
        ``score_partial`` draws during hypothetical insertion. The SHAP values approximate
        the feature contributions to the CoDisp-based anomaly score. The explanation operates
        in the shingled feature space (shape ``shingle_size * num_features``), not the original
        feature space.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score and explain.

        Returns:
            tuple: (score, shap_values) where
                score (float): The anomalousness score of the input instance (from ``score_partial``).
                shap_values (np.float64 array of shape (shingle_size * num_features,)): SHAP values
                in the shingled feature space. Positive values mean the feature increases the
                anomaly score (more anomalous), negative values mean it decreases the anomaly score
                (more normal). Returns zeros before any shingle is fitted.

        Raises:
            ValueError: If `X` holds NaN.
        """
        x = np.asarray(X)
        if x.dtype.kind in "fc" and bool(np.isnan(x).any()):
            raise ValueError(f"{type(self).__name__} does not accept NaN values.")

        score = self.score_partial(X)

        point = self._shingle(x)
        if point is None:
            return score, np.zeros(x.shape[0], dtype=np.float64)

        num_features = len(point)

        # Collect tree structures for all trees
        trees_features: list[NDArray[np.intp]] = []
        trees_thresholds: list[NDArray[np.float64]] = []
        trees_left: list[NDArray[np.intp]] = []
        trees_right: list[NDArray[np.intp]] = []
        trees_values: list[NDArray[np.float64]] = []
        trees_left_fractions: list[NDArray[np.float64]] = []

        for tree_id in range(self.num_trees):
            root = self._arena.roots[tree_id]
            if root < 0:
                continue
            feat, thresh, left, right, vals, left_fracs = self._collect_tree_structure(tree_id)
            trees_features.append(feat)
            trees_thresholds.append(thresh)
            trees_left.append(left)
            trees_right.append(right)
            trees_values.append(vals)
            trees_left_fractions.append(left_fracs)

        if not trees_features:
            return score, np.zeros(num_features, dtype=np.float64)

        shap_values = treeshap_ensemble(
            point,
            trees_features,
            trees_thresholds,
            trees_left,
            trees_right,
            trees_values,
            num_features,
            trees_left_fractions=trees_left_fractions,
        )

        return score, shap_values


# Tree cut and collusive displacement logic is adapted from rrcf 0.4.4, distributed under:
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
