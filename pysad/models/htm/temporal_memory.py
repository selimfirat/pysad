from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state


class TemporalMemory:
    """HTM sequence memory that predicts the next set of active columns :cite:`cui2016continuous`.

    The network has `num_columns` columns of `num_cells_per_column` cells. Each cell has distal segments, and each segment has synapses to other cells with a permanence in [0, 1]. A synapse is connected if its permanence is above `connected_permanence`. `fit` takes the active columns W^t of one time step and applies the rules of :cite:`cui2016continuous` (Section 3.3):

    1. Eq. 2: in each active column, the cells that were predicted become active. If no cell of the column was predicted, all its cells become active (the column bursts).
    2. Eq. 3: a segment that predicted an active cell is reinforced. Its synapses to cells active at t - 1 gain `permanence_increment`, and its other synapses lose `permanence_decrement`. In a bursting column, the segment with the most synapses to cells active at t - 1 is reinforced. If no segment matches, a new segment grows on the cell of the column with the fewest segments.
    3. Eq. 4: a segment that predicted a cell that did not become active loses `predicted_segment_decrement` on all its synapses.
    4. Eq. 1: a cell is predicted for the next step if one of its segments has more than `activation_threshold` connected synapses to the active cells.

    `score` gives the raw anomaly score of :cite:`ahmad2017unsupervised` (Eq. 1), the fraction of the active columns that contain no predicted cell. It does not change the state. The defaults are the parameters of :cite:`cui2016continuous` (Table 1).

    The papers leave these details open, and this implementation settles them as follows:

    - A segment in a bursting column matches if it has at least `learning_threshold` synapses to cells active at t - 1. Ties go to a fixed order of the segments. Ties between the cells with the fewest segments are broken at random.
    - Each segment that learns grows synapses to a random sample of the cells active at t - 1 that it has no synapse to, so that it has up to `max_new_synapses` synapses to them. A new segment starts with `initial_permanence` on each synapse. Because the paper samples from all active cells, not one winner cell per column, a bursting column at t - 1 gives the sample cells from all its cells.
    - A synapse whose permanence falls to 0 is removed. If a segment has no room for new synapses, its synapses with the lowest permanence are removed first. If a cell has `max_segments_per_cell` segments, the segment that learned least recently is removed before a new one grows.
    - If no column is active, the raw score is 0.

    Args:
        num_columns (int): Number of columns N (Default=2048).
        num_cells_per_column (int): Number of cells M in each column (Default=32).
        activation_threshold (int): A segment is active if it has more than this number of connected synapses to active cells, the threshold theta (Default=15).
        learning_threshold (int): Smallest number of synapses to cells active at t - 1 for a segment of a bursting column to match (Default=10).
        initial_permanence (float): Permanence of a new synapse (Default=0.21).
        connected_permanence (float): A synapse is connected if its permanence is above this value (Default=0.5).
        permanence_increment (float): Permanence increment p+ (Default=0.1).
        permanence_decrement (float): Permanence decrement p- (Default=0.1).
        predicted_segment_decrement (float): Permanence decrement p-- of segments whose cell was predicted but did not become active (Default=0.01).
        max_segments_per_cell (int): Maximum number of segments of one cell (Default=128).
        max_synapses_per_segment (int): Maximum number of synapses of one segment (Default=128).
        max_new_synapses (int): Maximum number of synapses that one segment has to the cells active at t - 1 after it learns (Default=32).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the sampled synapses and the cells of new segments. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self,
        num_columns: int = 2048,
        num_cells_per_column: int = 32,
        activation_threshold: int = 15,
        learning_threshold: int = 10,
        initial_permanence: float = 0.21,
        connected_permanence: float = 0.5,
        permanence_increment: float = 0.1,
        permanence_decrement: float = 0.1,
        predicted_segment_decrement: float = 0.01,
        max_segments_per_cell: int = 128,
        max_synapses_per_segment: int = 128,
        max_new_synapses: int = 32,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if max_new_synapses > max_synapses_per_segment:
            raise ValueError(
                f"max_new_synapses ({max_new_synapses}) must be at most max_synapses_per_segment ({max_synapses_per_segment})."
            )
        self.num_columns = num_columns
        self.num_cells_per_column = num_cells_per_column
        self.activation_threshold = activation_threshold
        self.learning_threshold = learning_threshold
        self.initial_permanence = initial_permanence
        self.connected_permanence = connected_permanence
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.predicted_segment_decrement = predicted_segment_decrement
        self.max_segments_per_cell = max_segments_per_cell
        self.max_synapses_per_segment = max_synapses_per_segment
        self.max_new_synapses = max_new_synapses
        self.random_state = random_state
        self._rng = check_random_state(random_state)

        num_cells = num_columns * num_cells_per_column
        capacity = 1024
        # Segment s belongs to cell _segment_cells[s] (-1 if the slot is free). Its synapse k goes
        # from cell _presynaptic[s, k] (-1 if the slot is empty) with permanence _permanences[s, k].
        self._segment_cells = np.full(capacity, -1, dtype=np.int64)
        self._last_learned = np.zeros(capacity, dtype=np.int64)
        self._presynaptic = np.full((capacity, max_synapses_per_segment), -1, dtype=np.int64)
        self._permanences = np.zeros((capacity, max_synapses_per_segment))
        self._free_segments = list(range(capacity - 1, -1, -1))
        self._num_segments = np.zeros(num_cells, dtype=np.int64)
        # Flat synapse ids s * max_synapses_per_segment + k, grouped by presynaptic cell, so that one
        # step reads only the synapses of the active cells.
        self._outgoing: list[list[int]] = [[] for _ in range(num_cells)]
        self._prev_active = np.zeros(num_cells, dtype=bool)
        self._cell_offsets = np.arange(num_cells_per_column, dtype=np.int64)
        self._iteration = 0

        self._active_cells = np.empty(0, dtype=np.int64)
        self._dirty_segments = np.empty(0, dtype=np.int64)
        self._active_segments = np.empty(0, dtype=np.int64)
        self._matching_segments = np.empty(0, dtype=np.int64)
        self._num_active_connected = np.zeros(capacity, dtype=np.int64)
        self._num_active_potential = np.zeros(capacity, dtype=np.int64)
        self._predicted_columns = np.empty(0, dtype=np.int64)

    def score(self, active_columns: np.ndarray) -> float:
        """Gives the raw anomaly score of the active columns of the next step without learning.

        Args:
            active_columns (np.int array of shape (num_active,)): Indices of the active columns.

        Returns:
            float: Fraction of the active columns that the previous step did not predict, in [0, 1].
        """
        active_columns = np.unique(active_columns)
        if active_columns.size == 0:
            return 0.0
        predicted = np.isin(active_columns, self._predicted_columns, assume_unique=True)
        return float(1.0 - predicted.sum() / active_columns.size)

    def fit(self, active_columns: np.ndarray) -> TemporalMemory:
        """Activates the cells of the active columns, learns the transition, and predicts the next step.

        Args:
            active_columns (np.int array of shape (num_active,)): Indices of the active columns.

        Returns:
            TemporalMemory: self.
        """
        M = self.num_cells_per_column
        active_columns = np.unique(active_columns)
        prev_active_cells = self._active_cells
        prev_active = self._prev_active
        prev_active[prev_active_cells] = True
        try:
            active_segments = self._active_segments
            segment_cells = self._segment_cells[active_segments]
            segment_columns = segment_cells // M
            correct = np.isin(segment_columns, active_columns)
            predicted_cells = np.unique(segment_cells[correct])
            bursting = np.setdiff1d(active_columns, segment_columns[correct], assume_unique=True)
            # (num_bursting, M) cells of the bursting columns.
            bursting_cells = bursting[:, None] * M + self._cell_offsets
            self._active_cells = np.concatenate([predicted_cells, bursting_cells.ravel()])

            # Eq. 4: punish the segments that predicted a cell of an inactive column.
            decrement = self.predicted_segment_decrement
            self._adapt(active_segments[~correct], prev_active, -decrement, decrement)

            learning = [active_segments[correct]]
            if bursting.size and prev_active_cells.size:
                matched, unmatched = self._best_matching(bursting)
                learning.append(matched)
                new_segments = self._create_segments(bursting_cells[unmatched])
                learning.append(new_segments)
            learning_segments = learning[0] if len(learning) == 1 else np.concatenate(learning)
            self._last_learned[learning_segments] = self._iteration

            # Eq. 3: reinforce the segments that predicted or best matched an active column.
            self._adapt(
                learning_segments,
                prev_active,
                self.permanence_increment,
                self.permanence_decrement,
            )
            if prev_active_cells.size:
                for segment in learning_segments.tolist():
                    self._grow_synapses(segment, prev_active_cells, prev_active)
        finally:
            prev_active[prev_active_cells] = False

        self._update_activity()
        self._iteration += 1
        return self

    def _best_matching(self, bursting: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Finds the segment with the most synapses to cells active at t - 1 in each bursting column."""
        M = self.num_cells_per_column
        candidates = self._matching_segments
        if candidates.size == 0:
            return np.empty(0, dtype=np.int64), np.ones(bursting.size, dtype=bool)
        columns = self._segment_cells[candidates] // M
        in_bursting = np.isin(columns, bursting)
        candidates, columns = candidates[in_bursting], columns[in_bursting]
        if candidates.size == 0:
            return np.empty(0, dtype=np.int64), np.ones(bursting.size, dtype=bool)
        # Sort by column, then by the number of active synapses (descending), then by segment id.
        order = np.lexsort((candidates, -self._num_active_potential[candidates], columns))
        candidates, columns = candidates[order], columns[order]
        first = np.ones(columns.size, dtype=bool)
        first[1:] = columns[1:] != columns[:-1]
        matched = candidates[first]
        unmatched = ~np.isin(bursting, columns[first], assume_unique=True)
        return matched, unmatched

    def _create_segments(self, cells: np.ndarray) -> np.ndarray:
        """Grows one segment in each row of cells, on the cell with the fewest segments."""
        if cells.size == 0:
            return np.empty(0, dtype=np.int64)
        counts = self._num_segments[cells]
        # Random tie break: the added noise is below 1, so it keeps the order of the counts.
        owners = cells[
            np.arange(cells.shape[0]),
            (counts + self._rng.random_sample(cells.shape)).argmin(axis=1),
        ]
        segments = np.empty(owners.size, dtype=np.int64)
        for i, cell in enumerate(owners.tolist()):
            if self._num_segments[cell] >= self.max_segments_per_cell:
                own = np.flatnonzero(self._segment_cells == cell)
                self._destroy_segment(int(own[self._last_learned[own].argmin()]))
            if not self._free_segments:
                self._grow_capacity()
            segment = self._free_segments.pop()
            self._segment_cells[segment] = cell
            self._last_learned[segment] = self._iteration
            self._num_segments[cell] += 1
            segments[i] = segment
        return segments

    def _destroy_segment(self, segment: int) -> None:
        slots = np.flatnonzero(self._presynaptic[segment] >= 0)
        self._remove_synapses(segment, slots)
        self._num_segments[self._segment_cells[segment]] -= 1
        self._segment_cells[segment] = -1
        self._num_active_connected[segment] = 0
        self._num_active_potential[segment] = 0
        self._free_segments.append(segment)

    def _grow_capacity(self) -> None:
        capacity = self._segment_cells.size
        self._segment_cells = np.concatenate(
            [self._segment_cells, np.full(capacity, -1, dtype=np.int64)]
        )
        self._last_learned = np.concatenate(
            [self._last_learned, np.zeros(capacity, dtype=np.int64)]
        )
        self._presynaptic = np.concatenate([self._presynaptic, np.full_like(self._presynaptic, -1)])
        self._permanences = np.concatenate([self._permanences, np.zeros_like(self._permanences)])
        self._num_active_connected = np.concatenate(
            [self._num_active_connected, np.zeros(capacity, dtype=np.int64)]
        )
        self._num_active_potential = np.concatenate(
            [self._num_active_potential, np.zeros(capacity, dtype=np.int64)]
        )
        self._free_segments.extend(range(2 * capacity - 1, capacity - 1, -1))

    def _adapt(
        self, segments: np.ndarray, prev_active: np.ndarray, increment: float, decrement: float
    ) -> None:
        """Adds `increment` to the synapses from cells active at t - 1 and subtracts `decrement` from the others."""
        if segments.size == 0:
            return
        presynaptic = self._presynaptic[segments]  # (num_segments, max_synapses_per_segment)
        exists = presynaptic >= 0
        delta = np.where(prev_active[presynaptic], increment, -decrement)
        permanences = np.where(exists, (self._permanences[segments] + delta).clip(0.0, 1.0), 0.0)
        self._permanences[segments] = permanences
        dead_rows, dead_slots = np.nonzero(exists & (permanences <= 0.0))
        if dead_rows.size:
            max_syn = self.max_synapses_per_segment
            for seg, slot in zip(segments[dead_rows].tolist(), dead_slots.tolist(), strict=True):
                cell = int(self._presynaptic[seg, slot])
                self._outgoing[cell].remove(seg * max_syn + slot)
                self._presynaptic[seg, slot] = -1
                self._permanences[seg, slot] = 0.0

    def _grow_synapses(
        self, segment: int, prev_active_cells: np.ndarray, prev_active: np.ndarray
    ) -> None:
        presynaptic = self._presynaptic[segment]
        exists = presynaptic >= 0
        existing = presynaptic[exists]
        active_existing = existing[prev_active[existing]]
        num_new = self.max_new_synapses - active_existing.size
        if num_new <= 0:
            return
        if active_existing.size == 0:
            candidates = prev_active_cells
        else:
            prev_active[active_existing] = False
            try:
                candidates = prev_active_cells[prev_active[prev_active_cells]]
            finally:
                prev_active[active_existing] = True
        if candidates.size > num_new:
            candidates = self._rng.choice(candidates, size=num_new, replace=False)
        excess = int(exists.sum()) + candidates.size - self.max_synapses_per_segment
        if excess > 0:
            weakest = np.flatnonzero(exists)[
                self._permanences[segment, exists].argsort(kind="stable")[:excess]
            ]
            self._remove_synapses(segment, weakest)
            exists[weakest] = False
        slots = np.flatnonzero(~exists)[: candidates.size]
        presynaptic[slots] = candidates
        self._permanences[segment, slots] = self.initial_permanence
        base = segment * self.max_synapses_per_segment
        for cell, slot in zip(candidates.tolist(), slots.tolist(), strict=True):
            self._outgoing[cell].append(base + slot)

    def _remove_synapses(self, segment: int, slots: np.ndarray) -> None:
        base = segment * self.max_synapses_per_segment
        for cell, slot in zip(
            self._presynaptic[segment, slots].tolist(), slots.tolist(), strict=True
        ):
            self._outgoing[cell].remove(base + slot)
        self._presynaptic[segment, slots] = -1
        self._permanences[segment, slots] = 0.0

    def _update_activity(self) -> None:
        """Counts the synapses of each segment to the active cells and predicts the next columns (Eq. 1)."""
        if self._dirty_segments.size:
            self._num_active_potential[self._dirty_segments] = 0
            self._num_active_connected[self._dirty_segments] = 0

        outgoing = self._outgoing
        flat: list[int] = []
        for cell in self._active_cells.tolist():
            if outgoing[cell]:
                flat.extend(outgoing[cell])

        if not flat and self.activation_threshold >= 0 and self.learning_threshold > 0:
            self._dirty_segments = np.empty(0, dtype=np.int64)
            self._active_segments = np.empty(0, dtype=np.int64)
            self._matching_segments = np.empty(0, dtype=np.int64)
            self._predicted_columns = np.empty(0, dtype=np.int64)
            return

        if flat:
            ids = np.asarray(flat, dtype=np.int64)
            segments = ids // self.max_synapses_per_segment
            pot_all = np.bincount(segments)
            unique_segs = np.flatnonzero(pot_all)
            pot_counts = pot_all[unique_segs]
            self._num_active_potential[unique_segs] = pot_counts
            self._dirty_segments = unique_segs

            connected = self._permanences.ravel()[ids] > self.connected_permanence
            conn_counts = np.bincount(segments[connected], minlength=pot_all.size)[unique_segs]
            self._num_active_connected[unique_segs] = conn_counts
        else:
            unique_segs = np.empty(0, dtype=np.int64)
            pot_counts = np.empty(0, dtype=np.int64)
            conn_counts = np.empty(0, dtype=np.int64)
            self._dirty_segments = unique_segs

        if self.activation_threshold >= 0:
            active_segs = unique_segs[conn_counts > self.activation_threshold]
        else:
            active_segs = np.flatnonzero(
                (self._segment_cells >= 0)
                & (self._num_active_connected > self.activation_threshold)
            )
        self._active_segments = active_segs

        if self.learning_threshold > 0:
            self._matching_segments = unique_segs[pot_counts >= self.learning_threshold]
        else:
            self._matching_segments = np.flatnonzero(
                (self._segment_cells >= 0) & (self._num_active_potential >= self.learning_threshold)
            )

        self._predicted_columns = np.unique(
            self._segment_cells[active_segs] // self.num_cells_per_column
        )
