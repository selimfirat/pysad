from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class SDOstream(BaseModel):
    """Scores each instance by its median distance to the nearest active observers :cite:`hartl2020low`.

    SDOstream extends the Sparse Data Observers (SDO) algorithm for stream data. The model holds `num_observers` points, called observers. It builds a low-density model of the stream and fades the quality of each observer with an exponential factor. The model follows the stream without a sliding window and uses a fixed amount of memory.

    Each observer holds a quality metric. The metric is the exponential moving average of how often the observer is one of the closest observers of a new instance, so recent uses count more than old uses. This class splits the observers into active and idle. The observers with the lowest quality, the `idle_fraction` fraction that the paper calls idle, stay out of the score.

    On each instance this class finds the `num_neighbors` closest active observers and returns the median of their distances. An instance far from every active observer gets a high score. Higher scores mean more anomalous instances.

    To keep the model current, this class adds new observers at a mean rate of `num_observers` points per unit of time `time_param`. The fading factor is exp(-1 / `time_param`). When the model is full, this class removes the observer with the lowest average quality. The average quality divides the quality metric by the highest value that the metric can reach at the age of the observer. A new observer then holds the highest average quality, so it does not get removed right away.

    The paper gives no rounding rule for the number of idle observers. This class follows the C++ reference implementation of the authors, dSalmon (LGPL-3.0), where the paper is silent. Where that implementation and the paper differ, this class follows the paper. The deviations from the paper:

    - PySAD passes no timestamps. This class uses the sample index as the time, with a constant inter-arrival time of one, so the time ratio of line 4 is one. The reference implementation raises the timestamps by one for each sample when it gets no timestamps. That ratio uses differences of times only, so the start value changes nothing.
    - Algorithm 1 fades the quality metric of every observer on each instance. This class stores the time of the last change of each quality value and fades the value when it reads it or changes it. Both approaches give the same values.
    - Algorithm 1 finds the idle-active split at line 1, before it fades the quality at line 2. This class fades the quality to the current time before it reads the split, as the reference implementation does.
    - Algorithm 1 raises the quality of the nearest observers at line 3 and removes the observer with the lowest average quality at line 6. The reference implementation picks that observer before the raise. This class follows the paper, so the raise can save an observer from removal.
    - The paper does not say how to round the number of idle observers. This class keeps active the floor((n - 1) * (1 - `idle_fraction`)) + 1 observers with the highest current quality, where n is the number of observers, so at least one observer stays active. The reference implementation uses the same active set.
    - The reference implementation averages the two middle distances with the higher index for an even `num_neighbors`, and it reads past the end of its distance array when two observers are active. This class returns the standard median, as the paper describes.
    - The paper describes an observer rate that falls during start-up. Algorithm 1 gets that behavior from the quality fraction of line 4, so this class adds no separate start-up rule.
    - The default values of `idle_fraction` and `num_neighbors` follow the reference implementation. The reference implementation has no default values for `num_observers` and `time_param`, so this class chooses them.

    Args:
        num_observers (int): The number of observers k in the model (Default=100).
        num_neighbors (int): The number x of closest observers for the median distance (Default=6).
        idle_fraction (float): The fraction q_id of observers to declare idle (Default=0.3).
        time_param (float): The time parameter T, with a fading factor of exp(-1 / T) (Default=250.0).
        random_state (int, np.random.RandomState or None): The seed or the random number generator of the observer sampling. None draws from the global NumPy random state, which `pysad.utils.fix_seed` seeds (Default=None).

    Attributes:
        observers_ (np.float64 array of shape (num_observers, num_features)): The points of the model. The first `size_` rows hold an observer.
        observations_ (np.float64 array of shape (num_observers,)): The quality metric of each observer, as of its last change.
        size_ (int): The number of observers in the model, at most `num_observers`.
        index_ (int): The number of instances that the model processed, which is the current time.
    """

    def __init__(
        self,
        num_observers: int = 100,
        num_neighbors: int = 6,
        idle_fraction: float = 0.3,
        time_param: float = 250.0,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.num_observers = num_observers
        self.num_neighbors = num_neighbors
        self.idle_fraction = idle_fraction
        self.time_param = time_param
        self.random_state = random_state

        # None is resolved on every draw, so a pickled model keeps drawing from the global
        # state rather than from a frozen copy, as pysad.models.RandomModel does.
        self._rng: np.random.RandomState | None = (
            None if random_state is None else check_random_state(random_state)
        )
        self._fading = float(np.exp(-1.0 / time_param))
        # The rate of a full model, the factor k^2 / x in front of the quality fraction of
        # Algorithm 1, line 4, times 1 / T because exp(-1 / T) gives one full replacement per T.
        self._rate = num_observers**2 / (num_neighbors * time_param)

        self.index_ = 0  # the sample index, used as time with a unit inter-arrival time
        self.size_ = 0  # the number of observers in the model
        self._to_init = True
        self.observers_ = np.empty((0, 0))  # (num_observers, num_features)
        # The quality metric of each observer, the value as of _touched (Algorithm 1's P_ω).
        self.observations_ = np.zeros(num_observers)
        # The time of the last change of each quality value, and the time each observer joined
        # the model, which is also its order for tie breaks.
        self._touched = np.zeros(num_observers, dtype=np.int64)
        self._added = np.zeros(num_observers, dtype=np.int64)

    def _faded(self, now: int) -> np.ndarray:
        """Returns the quality metric of each observer faded to the time `now`."""
        return self.observations_[: self.size_] * self._fading ** (
            now - self._touched[: self.size_]
        )

    def _squared_distances(self, X: np.ndarray) -> np.ndarray:
        """Returns the squared Euclidean distance from `X` to each observer."""
        diffs = self.observers_[: self.size_] - X
        return np.einsum("ij,ij->i", diffs, diffs)  # (size_,)

    def _closest(self, sq: np.ndarray, slots: np.ndarray) -> np.ndarray:
        """Returns the slots of the `num_neighbors` closest observers in `slots`.

        A tie on distance keeps the older observer, for reproducibility.
        """
        values = sq[slots]
        if values.size <= self.num_neighbors:
            return slots

        chosen = np.lexsort((self._added[slots], values))[: self.num_neighbors]
        return slots[chosen]

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> SDOstream:
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        X = np.asarray(X, dtype=np.float64)
        if self._to_init:
            self.observers_ = np.zeros((self.num_observers, X.shape[0]))
            self._to_init = False

        now = self.index_
        n = self.size_
        sq = self._squared_distances(X)  # (n,)
        faded = self._faded(now)  # (n,), the quality of each observer before the update

        # The num_neighbors closest observers N over all observers (Algorithm 1, line 1).
        nearest = self._closest(sq, np.arange(n))

        # The quality sums over all observers and over N, each raised by one at line 3.
        observations_sum = faded.sum() + nearest.size
        observations_nearest_sum = faded[nearest].sum() + nearest.size

        # With a unit inter-arrival time the time equals the index, so the time ratio of line 4
        # is one. The probability is then the rate times the fraction of the quality that N
        # holds. An empty model always keeps the first instance.
        add = n == 0
        if not add:
            rng = self._rng if self._rng is not None else check_random_state(None)
            add = rng.uniform() < self._rate * observations_nearest_sum / observations_sum

        # Line 3 raises the quality of each observer in N by one and ages it to now.
        faded[nearest] += 1.0
        self.observations_[nearest] = faded[nearest]
        self._touched[nearest] = now

        if add:
            if n == self.num_observers:
                # Line 6 removes the observer with the lowest average quality P̄_ω.
                age = 1.0 - self._fading ** (now - self._added[:n])
                slot = int(np.argmin(faded / age))
            else:
                slot = n
            self.observers_[slot] = X
            self.observations_[slot] = 1.0
            self._touched[slot] = now
            self._added[slot] = now
            if slot == n:
                self.size_ = n + 1

        self.index_ = now + 1
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        n = self.size_
        if n == 0:
            return 0.0  # the reference implementation scores the first instance 0

        X = np.asarray(X, dtype=np.float64)
        sq = self._squared_distances(X)  # (n,)
        faded = self._faded(self.index_)  # (n,), the current quality of each observer

        # The active observers hold the highest quality, so sort fading down and keep the top.
        # A tie on quality keeps the older observer, as the reference implementation does.
        order = np.lexsort((self._added[:n], -faded))
        active_count = int((n - 1) * (1.0 - self.idle_fraction)) + 1
        active = order[:active_count]  # the active observers

        # The num_neighbors closest active observers N_a (Algorithm 1, line 11).
        chosen = self._closest(sq, active)
        return float(np.median(np.sqrt(sq[chosen])))
