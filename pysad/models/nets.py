from __future__ import annotations

import numpy as np
import scipy

from pysad.core.base_model import BaseModel


class NETS(BaseModel):
    """Net-effect-based outlier detection with a cardinality grid (NETS) :cite:`yoon2019nets`, with the score of :class:`~pysad.models.ExactStorm`.

    An instance in the window of length `window_size` is a neighbor of the scored instance if their distance is not greater than `max_radius` (R). As in the paper, NETS partitions the space into hypercube cells with a diagonal of R, so the side length is R / sqrt(d) for d features, and it keeps the cardinality of each non-empty cell in the window. The cell of an instance comes from dividing each coordinate by the side length. All instances in the cell of the scored instance are its neighbors (Lemma 1), so the score counts them without distances. Its other neighbors are in the neighbor cells, whose centers are within 2R of the center of its cell (Definition 7 and Lemma 2). The score computes the distances only to the instances of these cells, as the point-level range search of the paper does.

    The paper reports an instance as an outlier if it has fewer than k neighbors. This method gives the score of :class:`~pysad.models.ExactStorm` instead: the fraction of window instances that are not neighbors of the scored instance. Thus the two models give the same scores. The cell-level bounds (Theorems 1 and 2), the conservative outer neighbor counts (Theorem 3) and the sub-dimensional filtering of Section 5 only decide the outlier status for a fixed k, so this method does not use k or these steps. It uses the full-dimensional cells, which is the choice of the paper when the sub-dimensions give no gain. In the paper, the window moves by a slide of S instances, and NETS applies the net effect of the new and the expired slide to the cell cardinalities. Here each instance gets its score when it arrives, so the slide holds one instance: the cell of the new instance gains one, and the cell of the expired instance loses one.

    Until the window is full, the fraction uses the instances that the window holds so far. An instance scored against an empty window gets the maximum score of 1. If ``max_radius`` is None, the radius comes from the first ``calibration_size`` instances, as in :class:`~pysad.models.ExactStorm`, and the cells are built again each time the radius changes. If the radius is 0, each cell holds equal instances only.

    Args:
        window_size (int): The number of instances in the window to score (Default=10000).
        max_radius (float or None): Maximum radius for the near instance selection. If None, the method sets it from the first instances (Default=None).
        calibration_size (int): The number of first instances that set the radius when ``max_radius`` is None (Default=256).
        outlier_fraction (float): The fraction of the first instances with no neighbor in the radius, when ``max_radius`` is None (Default=0.05).
    """

    def __init__(
        self,
        window_size: int = 10000,
        max_radius: float | None = None,
        calibration_size: int = 256,
        outlier_fraction: float = 0.05,
    ) -> None:
        self.window_size = window_size
        self.max_radius = max_radius
        self.calibration_size = calibration_size
        self.outlier_fraction = outlier_fraction
        self.radius_: float | None = max_radius
        self._calibration: list[np.ndarray] = []  # the first instances, while they set the radius
        self._nearest = np.empty(0)  # the distance from each of them to its nearest other one
        self.points_ = np.empty((0, 0))  # the window as a ring buffer, (window_size, num_features)
        self.cell_ = np.full(window_size, -1)  # the cell of each slot, or -1 for none
        self.cells_ = np.empty((0, 0), dtype=np.int64)  # the grid index of each cell, (window_size, num_features)
        self.card_ = np.zeros(window_size, dtype=np.int64)  # the cardinality grid, 0 is a free cell
        self._cells: dict[tuple[int, ...], int] = {}  # the cell of each non-empty grid index
        self._num = 0  # the number of instances in the window
        self._next = 0  # the slot of the next instance, which holds the oldest one in a full window

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> NETS:
        """Updates the model with the next instance. The oldest instance of a full window expires.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        X = np.array(X, dtype=np.float64)
        if self.points_.size == 0:
            self.points_ = np.empty((self.window_size, len(X)))
            self.cells_ = np.empty((self.window_size, len(X)), dtype=np.int64)

        i = self._next
        if self._num == self.window_size:
            self._remove(i)
        else:
            self._num += 1
        self.points_[i] = X
        self._next = (i + 1) % self.window_size

        if self.max_radius is None and len(self._calibration) < self.calibration_size:
            self._calibrate(X)
            if self.radius_ is not None:
                self._build()
        elif self.radius_ is not None:
            self._add(i)

        return self

    def _calibrate(self, X: np.ndarray) -> None:
        dists = np.linalg.norm(np.array(self._calibration) - X, axis=1) if self._calibration else []
        self._nearest = np.append(np.minimum(self._nearest, dists), np.min(dists, initial=np.inf))
        self._calibration.append(X)
        if len(self._calibration) >= 2:
            self.radius_ = float(np.quantile(self._nearest, 1 - self.outlier_fraction))

    def _index(self, X: np.ndarray) -> np.ndarray:
        """The grid index of the cell that holds each instance in X."""
        assert self.radius_ is not None
        if self.radius_ == 0:
            # A cell with a diagonal of 0 holds equal instances only, so the bits of X index it (-0.0 + 0.0 is 0.0).
            return (X + 0.0).view(np.int64)
        return np.floor(X / (self.radius_ / np.sqrt(X.shape[-1]))).astype(np.int64)

    def _build(self) -> None:
        """Builds the cardinality grid of the window with the current radius."""
        cells, cell, card = np.unique(
            self._index(self.points_[: self._num]),
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        self.cells_[: len(cells)] = cells
        self.card_[:] = 0
        self.card_[: len(cells)] = card
        self.cell_[: self._num] = cell.ravel()
        self._cells = {key: c for c, key in enumerate(map(tuple, cells.tolist()))}

    def _add(self, i: int) -> None:
        key = tuple(self._index(self.points_[i]).tolist())
        c = self._cells.get(key)
        if c is None:
            # A window of n instances has at most n - 1 other non-empty cells, so a free cell exists.
            c = int(np.flatnonzero(self.card_ == 0)[0])
            self._cells[key] = c
            self.cells_[c] = key
        self.card_[c] += 1
        self.cell_[i] = c

    def _remove(self, i: int) -> None:
        c = self.cell_[i]
        self.cell_[i] = -1
        if c < 0:
            return

        self.card_[c] -= 1
        if self.card_[c] == 0:
            del self._cells[tuple(self.cells_[c].tolist())]

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance as if it arrived, without adding it to the window.

        When the instance arrives, the oldest instance of a full window expires, so the instance is scored against the `window_size - 1` most recent instances. This gives the same score as `fit_score_partial`.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        X = np.asarray(X, dtype=np.float64)
        expiring = self._num == self.window_size
        num_window = self._num - expiring
        if num_window == 0 or self.radius_ is None:
            return 1.0

        # The oldest instance expires when X arrives.
        expired = self.cell_[self._next] if expiring else -1
        key = self._index(X)
        own = self._cells.get(tuple(key.tolist()))
        num_neighbors = 0 if own is None else self.card_[own] - (expired == own)

        # The squared distance between cell centers in side lengths, so 2R is 4 * num_features.
        active = np.flatnonzero(self.card_)
        offsets = self.cells_[active] - key  # (num_cells, num_features)
        dists = (offsets * offsets).sum(axis=1)
        near = np.zeros(self.window_size + 1, dtype=bool)  # by cell, the last entry is for -1
        near[active[(dists > 0) & (dists <= 4 * len(X))]] = True
        candidates = near[self.cell_]
        if expiring:
            candidates[self._next] = False
        num_neighbors += (scipy.spatial.distance.cdist(self.points_[candidates], [X]) <= self.radius_).sum()

        return float(1.0 - num_neighbors / num_window)
