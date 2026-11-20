from __future__ import annotations

from collections import Counter, deque

import numpy as np
from scipy.special import logsumexp

from pysad.core.base_model import BaseModel

# The bandwidth of the reference implementation in a dimension where the kernel centers coincide with the point.
_MIN_BANDWIDTH = 0.001


def _log_density(X: np.ndarray, centers: np.ndarray, weights: np.ndarray) -> float:
    """The log of the local density of Definition 2.2 at `X` from kernel `centers` whose `weights` sum to 1."""
    dists = np.abs(centers - X)  # (k, d)
    bandwidth = weights @ dists  # (d,)
    bandwidth[bandwidth <= 0.0] = _MIN_BANDWIDTH
    log_kernels = -0.5 * (dists / bandwidth) ** 2 - np.log(np.sqrt(2 * np.pi) * bandwidth)  # (k, d)
    # The product over the dimensions overflows or underflows in many dimensions, so sum the logs.
    return float(logsumexp(log_kernels.sum(axis=1), b=weights))


def _nearest(cell_array: np.ndarray, cell: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The indices of the `k` sorted cells nearest to `cell`, and the distances to all cells in cell lengths."""
    # Distances in cell lengths are exact, so equidistant cells tie, and the lowest cell wins a tie.
    dists = np.linalg.norm(cell_array - cell, axis=1)  # (num_cells,)
    return np.argsort(dists, kind="stable")[:k], dists


class _KernelCenter:
    """A kernel center with the local density estimate of its last update."""

    def __init__(self, cell: tuple[float, ...], center: np.ndarray) -> None:
        self.cell = np.array(cell)
        self.center = center
        self.error = 0.0  # the cumulative error E of Definition 4.5
        self.total_weight = 0.0  # of the nearest kernel centers at the last update
        self.max_dist = np.inf  # in cell lengths, to the num_neighbors-th nearest kernel center
        self.log_density = 0.0

    def update(
        self,
        cell_array: np.ndarray,
        centers: np.ndarray,
        weights: np.ndarray,
        num_neighbors: int,
    ) -> None:
        """Finds the nearest of the sorted cells and estimates the local density at the center."""
        nearest, dists = _nearest(cell_array, self.cell, num_neighbors)
        self.total_weight = float(weights[nearest].sum())
        # With fewer than num_neighbors kernel centers, any new kernel center would be a neighbor.
        self.max_dist = float(dists[nearest].max()) if len(nearest) == num_neighbors else np.inf
        self.error = 0.0
        self.log_density = _log_density(
            self.center, centers[nearest], weights[nearest] / self.total_weight
        )


class STARE(BaseModel):
    """Local outlier detection by stationary region skipping (STARE) :cite:`yoon2020ultrafast`.

    STARE partitions the space into a grid of cells whose diagonal length is ``cell_size`` (theta_R). The center of each nonempty cell is a kernel center, and the number of window instances in the cell is its weight (Definition 4.1). The window holds the last ``window_size`` instances in slides of ``slide_size`` instances. When a slide is complete, the window drops the oldest slide and the weights change by the net change of each cell between the new and the expired slide (Algorithm 2).

    The local density at a point is the weighted product-kernel estimate of Definition 2.2 over the ``num_neighbors`` (theta_K) nearest kernel centers of the cell of the point. The Gaussian kernel of each dimension has the bandwidth of the weighted mean distance from the point to these kernel centers in that dimension, as in the reference implementation. The score of an instance is the local outlier score of Definition 2.3, ``(mu - D(x)) / sigma``, where ``D(x)`` is its local density and ``mu`` and ``sigma`` are the mean and standard deviation of the local densities at the nearest kernel centers of its cell.

    STARE updates the local density of a kernel center only if its cumulative error of Definition 4.5 is above ``skip_threshold`` (gamma). The error adds up, over the slides since the last update, the absolute net changes of the cells within the distance to the ``num_neighbors``-th nearest kernel center, divided by the weight of the nearest kernel centers. A new kernel center is always updated. With ``skip_threshold`` = 0, every local density matches a direct computation on the window. The score of an instance always uses the current nearest kernel centers of its cell, as in Definition 2.3 and Algorithm 4: ``D(x)`` uses their current weights, and ``mu`` and ``sigma`` use their local densities, which for a skipped kernel center are from its last update.

    The deviations from the paper:

    - The paper reports the top-n outliers of each window, and it prunes the grid cells whose score bounds (Corollary 4.3) show that they hold no top-n outlier. This class returns the score of each instance, so it does not use the bounds.
    - ``score_partial`` scores an instance against the window of the last complete slide. The instances of a slide that is not complete yet are not part of the window. Before the first slide is complete, every instance gets the score 0. Until the window holds ``window_size`` instances, it holds the complete slides so far.
    - An instance whose cell has no kernel center uses the nearest kernel centers of the center of its cell.
    - The reference implementation anchors the grid at the minimum of each feature over the whole dataset, which a stream does not know in advance. This class anchors the grid at 0.
    - A kernel center that has fewer than ``num_neighbors`` kernel centers to choose from counts every net change in its cumulative error. A kernel center that is skipped keeps the density estimate of its last update, as in Theorem 4.7.
    - The paper does not say which kernel centers are the nearest when several cells are at the same distance, which often happens on a grid. This class picks the cell with the lowest index, in the order of the features. The reference implementation keeps the previous nearest kernel centers and otherwise follows the order of its hash map, so its results can differ, especially for a small ``num_neighbors``.
    - Definition 2.2 does not define the bandwidth of a dimension where the point coincides with all of its nearest kernel centers, for example a constant feature at the local density of a kernel center. The weighted mean distance there is 0. This class uses the bandwidth 0.001 instead, as the reference implementation does. Each kernel of such a dimension is then about 400, so the local densities and the score depend on this value, for example when a feature is constant.
    - The score of Definition 2.3 is undefined if the local densities at the nearest kernel centers are equal, for example while the window has a single kernel center. Such an instance gets the score 0.
    - The paper tunes ``num_neighbors`` and ``cell_size`` for each dataset. The default ``cell_size`` fits features scaled to the range from 0 to 1. The default ``slide_size`` is 5% of the window, as in the experiments of the paper, and the default ``skip_threshold`` is the value 0.1 that the paper recommends.

    Args:
        window_size (int): The number of instances in the window, a multiple of ``slide_size`` (Default=1000).
        slide_size (int): The number of instances in a slide (Default=50).
        num_neighbors (int): The number of nearest kernel centers, theta_K in the paper (Default=10).
        cell_size (float): The diagonal length of a grid cell, theta_R in the paper (Default=0.1).
        skip_threshold (float): The error allowance threshold gamma, above which the local density of a kernel center is updated (Default=0.1).
    """

    def __init__(
        self,
        window_size: int = 1000,
        slide_size: int = 50,
        num_neighbors: int = 10,
        cell_size: float = 0.1,
        skip_threshold: float = 0.1,
    ) -> None:
        if window_size % slide_size:
            raise ValueError(
                f"window_size must be a multiple of slide_size {slide_size}, got {window_size}."
            )
        self.window_size = window_size
        self.slide_size = slide_size
        self.num_neighbors = num_neighbors
        self.cell_size = cell_size
        self.skip_threshold = skip_threshold
        self.slides_: deque[Counter[tuple[float, ...]]] = deque()  # the cell counts of each slide
        self.weights_: Counter[tuple[float, ...]] = Counter()  # the weight of each nonempty cell
        self.kernels_: dict[tuple[float, ...], _KernelCenter] = {}
        self._slide: Counter[tuple[float, ...]] = Counter()  # the counts of the incomplete slide
        self._slide_length = 0
        self._cells: list[tuple[float, ...]] = []  # the nonempty cells in sorted order
        self._cell_array = np.empty((0, 0))  # the indices of the cells, (num_cells, num_features)
        self._centers = np.empty((0, 0))  # the kernel centers, (num_cells, num_features)
        self._weights = np.empty(0)  # the weights of the kernel centers, (num_cells,)

    def _cell(self, X: np.ndarray) -> tuple[float, ...]:
        return tuple(np.floor(X / (self.cell_size / np.sqrt(len(X)))).tolist())

    def _center(self, cell: tuple[float, ...]) -> np.ndarray:
        return (np.array(cell) + 0.5) * (self.cell_size / np.sqrt(len(cell)))

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> STARE:
        """Adds the next instance to the current slide, and slides the window when the slide is complete.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        self._slide[self._cell(np.asarray(X, dtype=np.float64))] += 1
        self._slide_length += 1
        if self._slide_length == self.slide_size:
            self._slide_window()
            self._slide, self._slide_length = Counter(), 0
        return self

    def _slide_window(self) -> None:
        """Runs the data distribution update and the stationary region skip of Algorithms 2 and 3."""
        expired = (
            self.slides_.popleft()
            if len(self.slides_) == self.window_size // self.slide_size
            else Counter()
        )
        self.slides_.append(self._slide)
        changes = self._slide.copy()
        changes.subtract(expired)
        changed = [cell for cell, change in changes.items() if change]
        for cell in changed:
            self.weights_[cell] += changes[cell]
            if not self.weights_[cell]:
                del self.weights_[cell]
                self.kernels_.pop(cell, None)
        if not changed:  # the window has the same weights, so no density changes
            return

        changed_cells = np.array(changed)  # (num_changed, d)
        abs_changes = np.abs([changes[cell] for cell in changed], dtype=np.float64)
        for kernel in self.kernels_.values():
            near = np.linalg.norm(changed_cells - kernel.cell, axis=1) <= kernel.max_dist
            kernel.error += abs_changes[near].sum() / kernel.total_weight  # Eq. 4

        self._cells = sorted(self.weights_)
        self._cell_array = np.array(self._cells)
        self._centers = (self._cell_array + 0.5) * (self.cell_size / np.sqrt(len(self._cells[0])))
        self._weights = np.array([self.weights_[cell] for cell in self._cells], dtype=np.float64)
        for cell in self._cells:
            if cell not in self.kernels_:
                self.kernels_[cell] = _KernelCenter(cell, self._center(cell))
            elif self.kernels_[cell].error <= self.skip_threshold:
                continue
            self.kernels_[cell].update(
                self._cell_array, self._centers, self._weights, self.num_neighbors
            )

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance against the window of the last complete slide.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if not self.kernels_:
            return 0.0

        X = np.asarray(X, dtype=np.float64)
        nearest, _ = _nearest(self._cell_array, np.array(self._cell(X)), self.num_neighbors)
        weights = self._weights[nearest]
        log_densities = np.append(
            [self.kernels_[self._cells[i]].log_density for i in nearest],
            _log_density(X, self._centers[nearest], weights / weights.sum()),
        )  # (k + 1,)
        # A z-score is unchanged when all densities scale by one factor, so scale the largest to 1.
        densities = np.exp(log_densities - log_densities.max())
        spread = densities[:-1].std()
        if spread == 0.0:
            return 0.0
        return float((densities[:-1].mean() - densities[-1]) / spread)
