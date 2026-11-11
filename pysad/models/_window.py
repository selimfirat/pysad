from __future__ import annotations

import numpy as np


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
