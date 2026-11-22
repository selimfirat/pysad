from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from pysad.core.base_model import BaseModel


class _DenStreamMicroCluster:
    """Micro-cluster for DenStream.

    A micro-cluster maintains a fading summary of the points assigned to it. The weight,
    center, and radius are computed with an exponential fading factor so that recent points
    count more than old points.
    """

    __slots__ = (
        "linear_sum",
        "squared_sum",
        "weight",
        "last_edit_time",
        "creation_time",
        "decaying_factor",
        "_center",
    )

    def __init__(
        self,
        x: NDArray[np.float64],
        timestamp: int,
        decaying_factor: float,
    ) -> None:
        self.linear_sum = x.copy()
        self.squared_sum = x * x
        self.weight = 1.0
        self.last_edit_time = timestamp
        self.creation_time = timestamp
        self.decaying_factor = decaying_factor
        self._center: NDArray[np.float64] | None = None

    def _fading_factor(self, timestamp: int) -> float:
        """Returns the fading factor for the given timestamp."""
        return float(2.0 ** (-self.decaying_factor * (timestamp - self.last_edit_time)))

    def _creation_fading_factor(self, timestamp: int) -> float:
        """Returns the fading factor from creation time to the given timestamp."""
        return float(2.0 ** (-self.decaying_factor * (timestamp - self.creation_time)))

    def calc_weight(self, timestamp: int) -> float:
        """Returns the faded weight at the given timestamp."""
        return self.weight * self._fading_factor(timestamp)

    def calc_creation_weight(self, timestamp: int) -> float:
        """Returns the faded weight from creation at the given timestamp."""
        return self.weight * self._creation_fading_factor(timestamp)

    @property
    def center(self) -> NDArray[np.float64]:
        """Returns the center of the micro-cluster (independent of timestamp)."""
        if self._center is None:
            self._center = self.linear_sum / self.weight
        return self._center

    def radius_with(self, x: NDArray[np.float64], timestamp: int) -> float:
        """Returns the radius the cluster would have if x were inserted, without mutating state."""
        ff = self._fading_factor(timestamp)
        faded_weight = self.weight * ff
        faded_ls = self.linear_sum * ff
        faded_ss = self.squared_sum * ff

        new_weight = faded_weight + 1.0
        new_ls = faded_ls + x
        new_ss = faded_ss + x * x

        inv_nw = 1.0 / new_weight
        # Radius squared = trace(covariance) = sum(E[x_i^2] - E[x_i]^2)
        # = sum(new_ss / new_weight) - sum((new_ls / new_weight)^2)
        mean = new_ls * inv_nw
        diff = np.sum(new_ss * inv_nw - mean * mean)
        return math.sqrt(max(0.0, diff))

    def insert(self, x: NDArray[np.float64], timestamp: int) -> None:
        """Inserts a point into the micro-cluster and updates its statistics."""
        ff = self._fading_factor(timestamp)
        self.linear_sum = self.linear_sum * ff + x
        self.squared_sum = self.squared_sum * ff + x * x
        self.weight = self.weight * ff + 1.0
        self.last_edit_time = timestamp
        self._center = None

    def merge(self, other: _DenStreamMicroCluster, timestamp: int) -> None:
        """Merges another micro-cluster into this one."""
        # Fade both clusters to the current timestamp
        ff_self = self._fading_factor(timestamp)
        ff_other = other._fading_factor(timestamp)

        self.linear_sum = self.linear_sum * ff_self + other.linear_sum * ff_other
        self.squared_sum = self.squared_sum * ff_self + other.squared_sum * ff_other
        self.weight = self.weight * ff_self + other.weight * ff_other
        self.last_edit_time = timestamp
        if self.creation_time > other.creation_time:
            self.creation_time = other.creation_time
        self._center = None


class DenStream(BaseModel):
    """DenStream density-based streaming anomaly detector :cite:`cao2006density`.

    DenStream maintains two sets of micro-clusters: potential core micro-clusters (p-micro-clusters)
    that represent dense regions, and outlier micro-clusters (o-micro-clusters) that represent
    sparse regions or noise. On each instance, the algorithm tries to merge the instance into the
    nearest p-micro-cluster; if that fails, it tries the nearest o-micro-cluster; if that also
    fails, it creates a new o-micro-cluster.

    The model has an initialization phase where it collects `init_samples` points and runs a
    DBSCAN-like clustering to create initial p-micro-clusters. After initialization, it processes
    points online.

    The anomaly score of an instance is based on its distance to the nearest micro-cluster and the
    type of micro-cluster it falls into:
    - If it merges into a p-micro-cluster: score is the normalized distance to the cluster center.
    - If it merges into an o-micro-cluster: score is the normalized distance plus a penalty.
    - If it creates a new o-micro-cluster: score is 1.0 (maximum anomaly).

    The model prunes micro-clusters periodically based on their faded weights. P-micro-clusters
    with weight below `beta * mu` are removed. O-micro-clusters with weight below a threshold `xi`
    (which depends on their age and the decaying factor) are removed.

    Micro-cluster prune rule (RFC):

    - Prune every `prune_interval_` time units (T_p). When not given, T_p is computed as
      ceil((1 / lambda) * ln((beta * mu) / (beta * mu - 1))), the number of time units after which
      a micro-cluster of weight beta * mu would fade below the core threshold.

    - P-micro-clusters: removed when their faded weight at the current time is below beta * mu.

    - O-micro-clusters: removed when their faded weight is below xi, where xi equals
      (2^(-lambda * (t - creation_time + T_p)) - 1) / (2^(-lambda * T_p) - 1) with t the current
      time. This keeps an o-micro-cluster only while its weight can still grow above the core
      threshold before the next prune, and prunes it once it is unlikely to become dense.

    Deviations from the paper:

    - The paper defines micro-cluster assignment (p-cluster, o-cluster, or new o-cluster) but not a
      scalar anomaly score. This implementation maps that assignment to a continuous score in
      [0.0, 1.5]: normalized distance for p-cluster merges, normalized distance plus a penalty for
      o-cluster merges, and 1.0 for new o-clusters. This mapping is a PySAD interface choice, not
      part of the paper.

    - The micro-cluster radius is the square root of the trace of the covariance (the root-sum of
      per-feature variances), a common interpretation of the paper's cluster radius.

    Args:
        decaying_factor (float): The fading factor lambda (0 < lambda <= 1). Controls the importance
            of historical data. Higher values mean faster forgetting. Default=0.05.
        epsilon (float): The maximum radius of a micro-cluster. Defines the epsilon neighborhood.
            Default=1.5.
        beta (float): Threshold multiplier for outlier micro-clusters relative to core micro-clusters.
            Must be in (0, 1]. Default=0.5.
        mu (float): Minimum weight for a core micro-cluster. Must satisfy beta * mu > 1.
            Default=3.0.
        init_samples (int): Number of samples to collect during initialization before starting
            the online phase. During initialization, a DBSCAN-like clustering is run on the
            buffered samples to create initial p-micro-clusters. Default=50.
        stream_speed (int): Number of samples per time unit. The timestamp advances every
            `stream_speed` samples. Default=1.
        prune_interval (int): The number of time units between pruning steps (T_p in the paper).
            If not given, computed from decaying_factor, beta, and mu as
            ceil((1/lambda) * log((beta*mu)/(beta*mu-1))). Default=None.

    Attributes:
        p_micro_clusters_ (list[_DenStreamMicroCluster]): The potential core micro-clusters.
        o_micro_clusters_ (list[_DenStreamMicroCluster]): The outlier micro-clusters.
        timestamp_ (int): The current time step.
        prune_interval_ (int): The interval between pruning steps.
    """

    __bibtex__ = """
@inproceedings{cao2006density,
  title={Density-Based Clustering over an Evolving Data Stream with Noise},
  author={Cao, Feng and Ester, Martin and Qian, Weining and Zhou, Aoying},
  booktitle={Proceedings of the Sixth SIAM International Conference on Data Mining},
  pages={328--339},
  year={2006},
  organization={SIAM}
}
"""

    _input_kind = "multivariate"
    _output_kind = "score"
    _has_decision_rule = False
    _is_detector = True

    def __init__(
        self,
        decaying_factor: float = 0.05,
        epsilon: float = 1.5,
        beta: float = 0.5,
        mu: float = 3.0,
        init_samples: int = 50,
        stream_speed: int = 1,
        prune_interval: int | None = None,
    ) -> None:
        if not (0.0 < decaying_factor <= 1.0):
            raise ValueError(f"decaying_factor must be in (0, 1], got {decaying_factor}.")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        if not (0.0 < beta <= 1.0):
            raise ValueError(f"beta must be in (0, 1], got {beta}.")
        if beta * mu <= 1.0:
            raise ValueError(
                f"beta * mu must be > 1, got beta={beta}, mu={mu}, product={beta * mu}."
            )
        if init_samples < 1:
            raise ValueError(f"init_samples must be >= 1, got {init_samples}.")
        if stream_speed < 1:
            raise ValueError(f"stream_speed must be >= 1, got {stream_speed}.")

        self.decaying_factor = decaying_factor
        self.epsilon = epsilon
        self.beta = beta
        self.mu = mu
        self.init_samples = init_samples
        self.stream_speed = stream_speed
        self.prune_interval = prune_interval

        self.timestamp_ = -1
        self.initialized_ = False
        self.p_micro_clusters_: list[_DenStreamMicroCluster] = []
        self.o_micro_clusters_: list[_DenStreamMicroCluster] = []
        self._init_buffer: list[NDArray[np.float64]] = []  # buffered points during initialization
        self._n_samples_seen = 0

        # Compute prune interval (T_p) if not provided
        if prune_interval is None:
            self.prune_interval_ = math.ceil(
                (1.0 / decaying_factor) * math.log((beta * mu) / (beta * mu - 1.0))
            )
        else:
            self.prune_interval_ = prune_interval

        # Thresholds
        self._core_threshold = beta * mu  # weight threshold for p-micro-clusters

    def _distance(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        """Returns the Euclidean distance between two points."""
        return float(np.linalg.norm(a - b))

    def _get_nearest_cluster(
        self, x: NDArray[np.float64], clusters: list[_DenStreamMicroCluster]
    ) -> tuple[int | None, _DenStreamMicroCluster | None, float]:
        """Returns the index, cluster, and distance of the nearest cluster to x.

        Ties on distance are broken by choosing the first cluster with minimum distance.
        """
        if not clusters:
            return None, None, float("inf")

        min_dist = float("inf")
        nearest_idx = None
        for i, cluster in enumerate(clusters):
            dist = self._distance(x, cluster.center)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        assert nearest_idx is not None
        return nearest_idx, clusters[nearest_idx], min_dist

    def _merge_into_cluster(
        self, x: NDArray[np.float64], cluster: _DenStreamMicroCluster, timestamp: int
    ) -> bool:
        """Attempts to merge x into cluster. Returns True if successful."""
        if cluster.radius_with(x, timestamp) <= self.epsilon:
            cluster.insert(x, timestamp)
            return True
        return False

    def _prune_clusters(self) -> None:
        """Removes micro-clusters with faded weight below thresholds."""
        # Prune p-micro-clusters
        self.p_micro_clusters_ = [
            c
            for c in self.p_micro_clusters_
            if c.calc_weight(self.timestamp_) >= self._core_threshold
        ]

        # Prune o-micro-clusters
        # xi = (2^(-lambda * (t - creation_time + T_p)) - 1) / (2^(-lambda * T_p) - 1)
        denom = 2.0 ** (-self.decaying_factor * self.prune_interval_) - 1.0
        if denom == 0.0:
            # Avoid division by zero when prune_interval is 0
            return

        new_o_clusters = []
        for cluster in self.o_micro_clusters_:
            creation_weight = cluster.calc_creation_weight(self.timestamp_ + self.prune_interval_)
            xi = (creation_weight - 1.0) / denom
            if cluster.calc_weight(self.timestamp_) >= xi:
                new_o_clusters.append(cluster)
        self.o_micro_clusters_ = new_o_clusters

    def _initialize_clusters(self) -> None:
        """Runs DBSCAN-like clustering on buffered points to create initial p-micro-clusters."""
        if len(self._init_buffer) < 2:
            # Not enough points for clustering, create o-micro-clusters
            for point in self._init_buffer:
                mc = _DenStreamMicroCluster(point, self.timestamp_, self.decaying_factor)
                self.o_micro_clusters_.append(mc)
            return

        # Simple DBSCAN-like clustering
        points = np.array(self._init_buffer)
        n_points = len(points)
        labels = np.full(n_points, -1, dtype=int)  # -1 = noise
        cluster_id = 0

        for i in range(n_points):
            if labels[i] != -1:
                continue  # Already processed

            # Find neighbors within epsilon
            distances = np.linalg.norm(points - points[i], axis=1)
            neighbors = np.where(distances <= self.epsilon)[0]

            if len(neighbors) < self.mu:
                # Not enough neighbors, mark as noise (will become o-micro-cluster)
                continue

            # Start new cluster
            labels[i] = cluster_id
            # Expand cluster
            seed_set = list(neighbors)
            for idx in seed_set:
                if labels[idx] == -1:
                    labels[idx] = cluster_id
                    # Find neighbors of this point
                    distances_idx = np.linalg.norm(points - points[idx], axis=1)
                    neighbors_idx = np.where(distances_idx <= self.epsilon)[0]
                    if len(neighbors_idx) >= self.mu:
                        # Add new neighbors to seed set
                        for n_idx in neighbors_idx:
                            if labels[n_idx] == -1:
                                labels[n_idx] = cluster_id
                                seed_set.append(n_idx)

            cluster_id += 1

        # Create micro-clusters from clusters
        for c_id in range(cluster_id):
            cluster_points = points[labels == c_id]
            # Create micro-cluster from cluster points
            mc = _DenStreamMicroCluster(cluster_points[0], self.timestamp_, self.decaying_factor)
            for point in cluster_points[1:]:
                mc.insert(point, self.timestamp_)
            self.p_micro_clusters_.append(mc)

        # Create o-micro-clusters from noise points
        noise_points = points[labels == -1]
        for point in noise_points:
            mc = _DenStreamMicroCluster(point, self.timestamp_, self.decaying_factor)
            self.o_micro_clusters_.append(mc)

    def fit_partial(self, X: NDArray[np.float64], y: int | None = None) -> DenStream:
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int or None): Ignored since the model is unsupervised (Default=None).

        Returns:
            DenStream: The model itself.
        """
        X = np.asarray(X, dtype=np.float64)

        self._n_samples_seen += 1

        # Control stream speed: advance timestamp every stream_speed instances
        if self._n_samples_seen % self.stream_speed == 0:
            self.timestamp_ += 1

        # Initialization phase: buffer points until we have init_samples
        if not self.initialized_:
            self._init_buffer.append(X)
            if len(self._init_buffer) >= self.init_samples:
                self._initialize_clusters()
                self.initialized_ = True
                self._init_buffer = []  # Clear buffer
            return self

        # Online phase: try to merge into nearest p-micro-cluster
        merged = False
        if self.p_micro_clusters_:
            idx, nearest_p, dist = self._get_nearest_cluster(X, self.p_micro_clusters_)
            if nearest_p is not None and self._merge_into_cluster(X, nearest_p, self.timestamp_):
                merged = True

        # Try to merge into nearest o-micro-cluster
        if not merged and self.o_micro_clusters_:
            idx, nearest_o, dist = self._get_nearest_cluster(X, self.o_micro_clusters_)
            if nearest_o is not None and self._merge_into_cluster(X, nearest_o, self.timestamp_):
                merged = True
                # Check if o-micro-cluster becomes a p-micro-cluster
                if nearest_o.calc_weight(self.timestamp_) > self._core_threshold:
                    assert idx is not None
                    self.o_micro_clusters_.pop(idx)
                    self.p_micro_clusters_.append(nearest_o)

        # Create new o-micro-cluster if not merged
        if not merged:
            mc = _DenStreamMicroCluster(X, self.timestamp_, self.decaying_factor)
            self.o_micro_clusters_.append(mc)

        # Periodic pruning
        if self.timestamp_ > 0 and self.timestamp_ % self.prune_interval_ == 0:
            self._prune_clusters()

        return self

    def score_partial(self, X: NDArray[np.float64]) -> float:
        """Scores the anomalousness of the next instance.

        The score is based on the distance to the nearest micro-cluster and its type:
        - Merges into p-micro-cluster: normalized distance (0 to 1)
        - Merges into o-micro-cluster: normalized distance + penalty (0.5 to 1.5)
        - Creates new o-micro-cluster: 1.0

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The anomaly score (higher means more anomalous).
        """
        X = np.asarray(X, dtype=np.float64)

        if not self.initialized_:
            # Not enough data to score, return 0
            return 0.0

        # Check distance to nearest p-micro-cluster
        min_p_dist = float("inf")
        nearest_p = None
        for cluster in self.p_micro_clusters_:
            dist = self._distance(X, cluster.center)
            if dist < min_p_dist:
                min_p_dist = dist
                nearest_p = cluster

        # Check distance to nearest o-micro-cluster
        min_o_dist = float("inf")
        nearest_o = None
        for cluster in self.o_micro_clusters_:
            dist = self._distance(X, cluster.center)
            if dist < min_o_dist:
                min_o_dist = dist
                nearest_o = cluster

        # Determine score based on where the point would fall
        if nearest_p is not None and min_p_dist <= self.epsilon:
            # Would merge into p-micro-cluster: normal, score is normalized distance
            return min_p_dist / self.epsilon

        if nearest_o is not None and min_o_dist <= self.epsilon:
            # Would merge into o-micro-cluster: potential outlier
            # Score is normalized distance plus penalty
            return 0.5 + 0.5 * (min_o_dist / self.epsilon)

        # Would create new o-micro-cluster: strong outlier
        return 1.0
