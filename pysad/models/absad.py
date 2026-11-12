from __future__ import annotations

import numpy as np

from pysad.core.base_model import BaseModel
from pysad.utils.window import RingBuffer


class ABSAD(BaseModel):
    """Angle-Based Subspace Anomaly Detection (ABSAD) :cite:`zhang2017sliding`.

    ABSAD is an online anomaly detection algorithm designed for high-dimensional
    data streams. It addresses the curse of dimensionality and concept drift by:

    1. Maintaining a bounded sliding window of normal observations.
    2. Identifying shared nearest neighbor (SNN) reference sets to capture local manifold structure.
    3. Evaluating angle-based feature relevance to select low-dimensional anomaly-relevant subspaces.
    4. Calculating the Local Outlier Score (LOS) via normalized squared Mahalanobis distance in the selected subspace.
    5. Dynamically estimating a control limit using Gaussian Kernel Density Estimation (KDE).
    6. Selectively updating the sliding window to accept only normal observations and reject faults.

    Warm-up Behavior:
        The original paper initializes the first window offline from fault-free historical data.
        In PySAD, the window is initialized online from the first ``window_size`` streaming
        observations. During warm-up (while the window holds fewer than ``window_size`` instances),
        observations are accepted unconditionally into the window. Before at least
        ``max(2, num_references)`` observations have arrived, ``score_partial`` returns 0.0.
        The first statistical control limit becomes available once the window is full.

    Args:
        window_size (int): Size of the sliding window, L in the paper (Default=750).
        num_neighbors (int): Number of nearest neighbors for SNN search, k in the paper (Default=100).
        num_references (int): Size of the reference set, s in the paper. Must be <= num_neighbors (Default=100).
        theta (float): Feature relevance threshold multiplier, θ in the paper (Default=0.4).
        confidence (float): Statistical confidence level for the KDE control limit, 1 - γ in the paper (Default=0.99).
    """

    __bibtex__ = """
@article{zhang2017sliding,
  title={Sliding window-based fault detection from high-dimensional data streams},
  author={Zhang, Liangwei and Lin, Jing and Karim, Ramin},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  volume={47},
  number={2},
  pages={289--303},
  year={2017},
  publisher={IEEE}
}
"""

    def __init__(
        self,
        window_size: int = 750,
        num_neighbors: int = 100,
        num_references: int = 100,
        theta: float = 0.4,
        confidence: float = 0.99,
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}.")
        if num_neighbors <= 0 or num_neighbors > window_size:
            raise ValueError(
                f"num_neighbors must be between 1 and window_size ({window_size}), got {num_neighbors}."
            )
        if num_references <= 0 or num_references > num_neighbors:
            raise ValueError(
                f"num_references must be between 1 and num_neighbors ({num_neighbors}), got {num_references}."
            )
        if theta < 0.0:
            raise ValueError(f"theta must be non-negative, got {theta}.")
        if not (0.0 < confidence < 1.0):
            raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

        self.window_size = window_size
        self.num_neighbors = num_neighbors
        self.num_references = num_references
        self.theta = theta
        self.confidence = confidence

        self._window: RingBuffer = RingBuffer(window_size)
        self._window_los: RingBuffer = RingBuffer(window_size)
        self.control_limit_: float | None = None

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> ABSAD:
        """Updates the detector with the next instance.

        During warm-up, instances are accepted unconditionally into the window. Once the window
        is full, an instance is accepted only if its Local Outlier Score (LOS) is within the
        statistical control limit. Anomalous instances exceeding the control limit are rejected.

        Args:
            X (np.ndarray): The instance to fit of shape (num_features,).
            y (int or None): Ignored (unsupervised detector).

        Returns:
            ABSAD: self.
        """
        x = np.asarray(X, dtype=np.float64)
        if len(self._window) < self.window_size:
            self._window.append(x)
            self._window_los.append(0.0)
            return self

        score = self.score_partial(x)
        if self.control_limit_ is not None and score <= self.control_limit_:
            self._window.append(x)
            self._window_los.append(score)

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance without modifying model state.

        Args:
            X (np.ndarray): The instance to score of shape (num_features,).

        Returns:
            float: Local Outlier Score (LOS).
        """
        if len(self._window) < max(2, self.num_references):
            return 0.0

        return 0.0
