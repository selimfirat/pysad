from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class GNGA(BaseModel):
    """Tracks a growing neural gas graph for evolving streams :cite:`bouguelia2018adaptive`.

    GNG-A moves prototypes with error-ranked learning rates. It removes stale
    prototypes with an adaptive survival threshold and inserts new prototypes
    with a probability based on recent winning frequencies.

    The score is the distance to the nearest prototype minus the mean edge
    length. Positive scores mark anomalies under the paper's decision rule.
    This continuous margin replaces the paper's binary decision. Before the
    first fit, the score is zero. With one prototype, the score is its distance.
    If all edges have zero length, the score remains the nearest distance.

    This class follows Algorithms 2--4 in the published paper. Algorithm 2
    adds squared second-nearest distance to relevance, unlike Equation 2.
    This class follows Algorithm 2. Error ranks start at zero.
    Equal errors and distances keep slot order. Neighbor rates use the minimum
    of their own rate and the winner's rate, as Section 3 specifies.

    The paper leaves several startup details open. The first two instances
    initialize two linked prototypes without other updates. Time counts both instances.
    New prototypes start
    with zero wins, relevance one, and zero error. Inserted prototypes instead
    start with half their parent's error. Insertion splits the parent edge
    into two edges of age zero. Parent errors stay unchanged, as Algorithm 4
    specifies. The initial threshold is uniform on (0, 1].

    Removed prototypes keep fixed positions. Their nearest prototype wins
    separately within the removal buffer on each update. Its relevance uses
    the nearest active distance as the replacement cost. Both sets fade their
    error and relevance by 0.9. This resolves the buffer update rule in footnote 2.
    Relevance uses logarithms to prevent underflow during long inactive periods.

    Deviations that bound memory: the graph holds at most `max_neurons`
    prototypes. At capacity, creation stops until a removal frees space.
    The removal buffer also holds at most `max_neurons` prototypes and discards
    its oldest entry at capacity. Removal keeps at least two connected
    prototypes. These limits also cover constant streams and graph collapse,
    which the paper does not specify. The winning window starts at ten and
    grows by ten when the graph reaches its size, as Section 4.2 describes.
    Its maximum is `max_neurons` rounded up to the next multiple of ten.
    Searches found no public reference implementation.

    Args:
        k (int): The paper's stability parameter for removal and creation (Default=10).
        max_age (int): The maximum edge age, measured in wins of either endpoint (Default=10).
        max_neurons (int): The maximum number of active or buffered prototypes, at least two (Default=100).
        random_state (int, np.random.RandomState or None): The random seed or generator. None uses NumPy's global state (Default=None).

    Attributes:
        threshold_ (float): The adaptive survival threshold tau.
        window_size_ (int): The current winning window length W.
    """

    def __init__(
        self,
        k: int = 10,
        max_age: int = 10,
        max_neurons: int = 100,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if max_neurons < 2:
            raise ValueError("max_neurons must be at least two.")
        if k < 1:
            raise ValueError("k must be at least one.")
        self.k = k
        self.max_age = max_age
        self.max_neurons = max_neurons
        self.random_state = random_state
        self._rng = None if random_state is None else check_random_state(random_state)
        capacity = 2 * max_neurons
        self._positions = np.empty((capacity, 0))
        self._active = np.zeros(capacity, dtype=bool)
        self._buffer: deque[int] = deque()
        self._errors = np.zeros(capacity)
        self._log_relevance = np.zeros(capacity)
        self._ages = np.full((capacity, capacity), -1, dtype=np.int64)
        self._max_window = (max_neurons // 10 + 1) * 10
        self._wins = np.full((capacity, self._max_window), -self._max_window, dtype=np.int64)
        self._time = 0
        self._last_creation = 0
        self._threshold_updates = 1
        self.threshold_ = 0.0
        self.window_size_ = 10

    @property
    def neurons_(self) -> NDArray[np.float64]:
        """Returns active vectors with shape ``(num_neurons_, num_features)``."""
        active: NDArray[np.float64] = self._positions[self._active]
        return active

    @property
    def num_neurons_(self) -> int:
        """Returns the current graph size."""
        return int(self._active.sum())

    def _add(self, X: NDArray[np.float64]) -> int:
        """Adds a prototype in an unused slot."""
        occupied = self._active.copy()
        occupied[list(self._buffer)] = True
        slot = int(np.flatnonzero(~occupied)[0])
        self._positions[slot] = X
        self._active[slot] = True
        self._errors[slot] = 0.0
        self._log_relevance[slot] = 0.0
        self._wins[slot] = -self._max_window
        self._ages[slot] = -1
        self._ages[:, slot] = -1
        return slot

    def _discard(self, slots: NDArray[np.int64]) -> None:
        """Removes prototypes and their incident edges."""
        self._active[slots] = False
        self._ages[slots] = -1
        self._ages[:, slots] = -1

    def _remove(self, X: NDArray[np.float64], distance: float, counts: NDArray[np.int64]) -> None:
        """Applies Algorithm 3 to the graph and removal buffer."""
        rate = 1.0 / (1 + self._threshold_updates)
        if self._buffer:
            buffered = np.array(self._buffer)
            diffs = self._positions[buffered] - X
            sq = np.einsum("ij,ij->i", diffs, diffs)
            nearest = int(buffered[sq.argmin()])
            self._wins[nearest, self._time % self._max_window] = self._time
            counts[nearest] += 1
            self._errors[nearest] += sq.min()
            self._log_relevance[nearest] = np.logaddexp(
                self._log_relevance[nearest], np.log(max(distance, np.finfo(float).tiny))
            )
            if sq.min() < distance and -self._log_relevance[nearest] > self.threshold_:
                self.threshold_ += rate * (-self._log_relevance[nearest] - self.threshold_)
                self._threshold_updates += 1
            if np.count_nonzero(counts[buffered] == 0) > self.k:
                self._buffer.popleft()

        active = np.flatnonzero(self._active)
        candidate = int(active[self._log_relevance[active].argmin()])
        survival = -self._log_relevance[candidate]
        if np.count_nonzero(counts[active] == 0) > self.k and survival <= self.threshold_:
            self.threshold_ += rate * (survival - self.threshold_)
            self._threshold_updates += 1
        if survival <= self.threshold_:
            return

        # Keep a connected pair so the next instance has two nearest prototypes.
        remaining = active[active != candidate]
        linked = (self._ages[np.ix_(remaining, remaining)] >= 0).any(axis=1)
        if np.count_nonzero(linked) < 2:
            return
        if len(self._buffer) == self.max_neurons:
            self._buffer.popleft()
        self._buffer.append(candidate)
        self._discard(np.concatenate(([candidate], remaining[~linked])))

    def _create(self, counts: NDArray[np.int64]) -> None:
        """Applies the insertion probability and edge split of Algorithm 4."""
        active = np.flatnonzero(self._active)
        if active.size >= self.max_neurons:
            return
        top = np.sort(counts[active])[-self.k :]
        probability = float(top.mean()) / self.window_size_ - 1.0 / (
            self._time - self._last_creation
        )
        rng = self._rng if self._rng is not None else check_random_state(None)
        if rng.uniform() >= probability:
            return
        parent = int(active[self._errors[active].argmax()])
        neighbors = np.flatnonzero(self._ages[parent] >= 0)
        neighbor = int(neighbors[self._errors[neighbors].argmax()])
        slot = self._add(0.5 * self._positions[parent] + 0.5 * self._positions[neighbor])
        self._errors[slot] = 0.5 * self._errors[parent]
        self._ages[parent, neighbor] = self._ages[neighbor, parent] = -1
        self._ages[parent, slot] = self._ages[slot, parent] = 0
        self._ages[neighbor, slot] = self._ages[slot, neighbor] = 0
        self._last_creation = self._time

    def fit_partial(self, X: NDArray[np.float64], y: int | None = None) -> GNGA:
        """Learns from one stream instance.

        Args:
            X (np.ndarray): The instance, with shape (num_features,).
            y (int or None): Ignored (Default=None).

        Returns:
            GNGA: This model.
        """
        X = np.asarray(X, dtype=np.float64)
        self._time += 1
        if self._positions.shape[1] == 0:
            self._positions = np.empty((2 * self.max_neurons, X.size))
            rng = self._rng if self._rng is not None else check_random_state(None)
            self.threshold_ = 1.0 - rng.uniform()
        if self.num_neurons_ < 2:
            slot = self._add(X)
            if self.num_neurons_ == 2:
                self._ages[0, slot] = self._ages[slot, 0] = 0
            return self

        active = np.flatnonzero(self._active)
        if active.size >= self.window_size_:
            self.window_size_ += 10
        diffs = self._positions[active] - X
        sq = np.einsum("ij,ij->i", diffs, diffs)
        order = np.argsort(sq, kind="stable")
        winner, second = active[order[:2]]
        self._errors[winner] += sq[order[0]]
        self._log_relevance[winner] = np.logaddexp(
            self._log_relevance[winner], np.log(max(sq[order[1]], np.finfo(float).tiny))
        )
        column = self._time % self._max_window
        self._wins[:, column] = -self._max_window
        self._wins[winner, column] = self._time
        ranks = np.empty(active.size, dtype=np.int64)
        ranks[np.argsort(-self._errors[active], kind="stable")] = np.arange(active.size)
        rates = 1.0 / (2 + ranks)
        winner_rate = rates[order[0]]
        neighbors = self._ages[winner, active] >= 0
        moving = neighbors | (active == winner)
        self._positions[active[moving]] -= (
            np.minimum(winner_rate, rates[moving])[:, None] * diffs[moving]
        )
        self._ages[winner, active[neighbors]] += 1
        self._ages[active[neighbors], winner] += 1
        self._ages[winner, second] = self._ages[second, winner] = 0
        self._ages[self._ages > self.max_age] = -1
        self._discard(active[~(self._ages[active] >= 0).any(axis=1)])
        counts = (self._wins > self._time - self.window_size_).sum(axis=1)
        winner_diff = self._positions[winner] - X
        self._remove(X, float(winner_diff @ winner_diff), counts)
        self._create(counts)
        occupied = np.concatenate((np.flatnonzero(self._active), list(self._buffer))).astype(int)
        self._errors[occupied] *= 0.9
        self._log_relevance[occupied] += np.log(0.9)
        return self

    def score_partial(self, X: NDArray[np.float64]) -> float:
        """Returns the nearest distance minus the mean edge length.

        Args:
            X (np.ndarray): The instance, with shape (num_features,).

        Returns:
            float: The anomaly margin. Higher values mean more anomalous instances.
        """
        active = np.flatnonzero(self._active)
        if active.size == 0:
            return 0.0
        distance = np.linalg.norm(self._positions[active] - X, axis=1).min()
        left, right = np.nonzero(np.triu(self._ages >= 0, k=1))
        if left.size == 0:
            return float(distance)
        lengths = np.linalg.norm(self._positions[left] - self._positions[right], axis=1)
        return float(distance - lengths.mean())
