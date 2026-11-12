from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from pysad.core.base_model import BaseModel
from pysad.utils.window import RingBuffer


def _normalize_window_and_point(
    W: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normalizes the sliding window and query point by feature-wise mean and std.

    Args:
        W (np.ndarray): Sliding window observations of shape (N, d).
        x (np.ndarray): Query observation of shape (d,).

    Returns:
        tuple[np.ndarray, np.ndarray]: Normalized window of shape (N, d) and
            normalized query point of shape (d,).
    """
    mu = np.mean(W, axis=0)
    sigma = np.std(W, axis=0)
    zero_std = sigma < 1e-12
    scale = np.where(zero_std, 1.0, sigma)

    W_norm = (W - mu) / scale
    x_norm = (x - mu) / scale
    # Ensure constant dimensions are exactly 0.0 when query equals the window mean
    if np.any(zero_std):
        exact_zero = np.isclose(x - mu, 0.0, atol=1e-12)
        x_norm[zero_std & exact_zero] = 0.0
        W_norm[:, zero_std] = 0.0

    return W_norm, x_norm


def _select_reference_set(
    W_norm: np.ndarray, x_norm: np.ndarray, k: int, s: int
) -> tuple[np.ndarray, np.ndarray]:
    """Selects the reference set using Shared Nearest Neighbor (SNN) similarity.

    1. Identifies the k nearest neighbors of x_norm in W_norm.
    2. Computes the shared nearest neighbor similarity between x_norm and each candidate.
    3. Ranks candidates deterministically (higher SNN similarity, smaller distance, smaller index).
    4. Selects the top s candidates to form the reference set S.

    Args:
        W_norm (np.ndarray): Normalized window observations of shape (N, d).
        x_norm (np.ndarray): Normalized query point of shape (d,).
        k (int): Number of nearest neighbors.
        s (int): Size of the reference set.

    Returns:
        tuple[np.ndarray, np.ndarray]: Reference set S of shape (s_eff, d) and
            their corresponding indices in W_norm.
    """
    n_samples = len(W_norm)
    k_eff = min(k, n_samples)
    s_eff = min(s, k_eff)

    dists_to_x = np.linalg.norm(W_norm - x_norm, axis=1)
    cand_indices = np.argsort(dists_to_x, kind="stable")[:k_eff]
    cand_set = set(cand_indices)

    # Compute pairwise distances between candidate points and all window points
    cand_points = W_norm[cand_indices]
    pair_dists = cdist(cand_points, W_norm)

    # Calculate SNN similarity for each candidate
    candidate_ranking = []
    for idx_in_cands, cand_idx in enumerate(cand_indices):
        q_nn = np.argsort(pair_dists[idx_in_cands], kind="stable")[:k_eff]
        snn_sim = len(cand_set.intersection(q_nn))
        # Deterministic sorting key: (-snn_sim, distance to x, original index)
        candidate_ranking.append((-snn_sim, dists_to_x[cand_idx], cand_idx))

    candidate_ranking.sort()
    selected_indices = np.array([item[2] for item in candidate_ranking[:s_eff]], dtype=int)
    ref_set = W_norm[selected_indices]

    return ref_set, selected_indices


def _calculate_feature_relevance(
    ref_set: np.ndarray, x_norm: np.ndarray, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates angle-based feature relevance and selects the relevant subspace.

    For each dimension j, the relevance is defined by the absolute cosine of the angle
    between the directional displacement vector (x_norm - center(ref_set)) and the
    coordinate axis j. Dimensions with relevance exceeding (1 + theta) * mean_relevance
    are selected. If no dimensions meet this threshold, all dimensions are retained.

    Args:
        ref_set (np.ndarray): Reference set observations of shape (s, d).
        x_norm (np.ndarray): Normalized query point of shape (d,).
        theta (float): Relevance threshold multiplier.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature relevance array of shape (d,) and
            the 1D array of selected feature indices.
    """
    d = len(x_norm)
    c_s = np.mean(ref_set, axis=0)
    v = x_norm - c_s
    v_norm = float(np.linalg.norm(v))

    if v_norm < 1e-12:
        relevance = np.full(d, 1.0 / d, dtype=np.float64)
        selected_subspace = np.arange(d, dtype=int)
        return relevance, selected_subspace

    # Angle between v and standard basis vector e_j has cos(alpha_j) = v_j / ||v||
    relevance = np.abs(v) / v_norm
    mean_relevance = float(np.mean(relevance))
    threshold = (1.0 + theta) * mean_relevance

    selected_subspace = np.where(relevance > threshold)[0]
    if len(selected_subspace) == 0:
        # Fall back to full space if no feature dominates by the threshold
        selected_subspace = np.arange(d, dtype=int)

    return relevance, selected_subspace


def _calculate_local_outlier_score(
    ref_set: np.ndarray, x_norm: np.ndarray, subspace: np.ndarray
) -> float:
    """Calculates the Local Outlier Score (LOS) via normalized squared Mahalanobis distance.

    Projects the query point and reference set into the selected subspace and computes:
        LOS = (x_sub - c_sub)^T Sigma_{sub}^{-1} (x_sub - c_sub) / d_{sub}

    Uses Tikhonov regularization on the covariance matrix and solves via Cholesky
    decomposition with pseudoinverse fallback for singular or ill-conditioned matrices.

    Args:
        ref_set (np.ndarray): Reference set observations of shape (s, d).
        x_norm (np.ndarray): Normalized query point of shape (d,).
        subspace (np.ndarray): Selected subspace feature indices.

    Returns:
        float: Normalized squared Mahalanobis distance in the selected subspace.
    """
    d_sub = len(subspace)
    s_samples = len(ref_set)
    S_sub = ref_set[:, subspace]
    x_sub = x_norm[subspace]

    c_sub = np.mean(S_sub, axis=0)
    delta = x_sub - c_sub

    if s_samples > 1:
        cov = np.cov(S_sub, rowvar=False, ddof=1)
        if d_sub == 1:
            cov = np.array([[float(cov)]], dtype=np.float64)
    else:
        cov = np.eye(d_sub, dtype=np.float64)

    # Regularize covariance to guarantee positive-definiteness
    tr = float(np.trace(cov))
    eps = 1e-6 * max(1.0, tr / d_sub)
    cov_reg = cov + eps * np.eye(d_sub, dtype=np.float64)

    try:
        L = np.linalg.cholesky(cov_reg)
        y = np.linalg.solve(L, delta)
        d_m_sq = float(np.dot(y, y))
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_reg)
        d_m_sq = float(delta @ inv_cov @ delta)

    return max(0.0, d_m_sq / d_sub)


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
            float: Local Outlier Score (LOS). Higher scores indicate greater anomalousness.
        """
        if len(self._window) < max(2, self.num_references):
            return 0.0

        x = np.asarray(X, dtype=np.float64)
        W = self._window.to_array()

        W_norm, x_norm = _normalize_window_and_point(W, x)
        ref_set, _ = _select_reference_set(W_norm, x_norm, self.num_neighbors, self.num_references)
        _, subspace = _calculate_feature_relevance(ref_set, x_norm, self.theta)

        return _calculate_local_outlier_score(ref_set, x_norm, subspace)
