from __future__ import annotations

import numpy as np


class _RingBuffer:
    """Fixed-capacity 2D float64 ring buffer of recent feature vectors."""

    def __init__(self, window_size: int) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        self.window_size = window_size
        self.points = np.empty((0, 0), dtype=np.float64)
        self.count = 0
        self.next_slot = 0

    def append(self, X: np.ndarray) -> int:
        """Writes `X` into the next slot, advances the ring, and returns the written slot index."""
        if self.points.size == 0:
            self.points = np.empty((self.window_size, len(X)), dtype=np.float64)
        slot = self.next_slot
        self.points[slot] = X
        self.next_slot = (slot + 1) % self.window_size
        if self.count < self.window_size:
            self.count += 1
        return slot

    def active_slices(self) -> tuple[np.ndarray, np.ndarray | None, int]:
        """Returns zero-copy slice(s) of the window after the oldest slot expires, and their count."""
        if self.count < self.window_size:
            return self.points[: self.count], None, self.count
        slot = self.next_slot
        return self.points[:slot], self.points[slot + 1 :], self.window_size - 1


class _RadiusCalibrator:
    """Online nearest-neighbor quantile radius calibrator for distance-based detectors."""

    def __init__(
        self,
        max_radius: float | None,
        calibration_size: int,
        outlier_fraction: float,
    ) -> None:
        self.max_radius = max_radius
        self.calibration_size = calibration_size
        self.outlier_fraction = outlier_fraction
        self.radius: float | None = max_radius
        self.count = 0
        self._points = np.empty((0, 0), dtype=np.float64)
        self._nearest = np.empty(0, dtype=np.float64)

    @property
    def active(self) -> bool:
        return self.max_radius is None and self.count < self.calibration_size

    def update(self, X: np.ndarray) -> float | None:
        if self.count == 0:
            self._points = np.empty((self.calibration_size, len(X)), dtype=np.float64)
            self._nearest = np.full(self.calibration_size, np.inf, dtype=np.float64)
        else:
            dists = np.linalg.norm(self._points[: self.count] - X, axis=1)
            np.minimum(self._nearest[: self.count], dists, out=self._nearest[: self.count])
            self._nearest[self.count] = dists.min()
        self._points[self.count] = X
        self.count += 1
        if self.count >= 2:
            self.radius = float(np.quantile(self._nearest[: self.count], 1 - self.outlier_fraction))
        if self.count == self.calibration_size:
            self._points = np.empty((0, 0), dtype=np.float64)
            self._nearest = np.empty(0, dtype=np.float64)
        return self.radius
