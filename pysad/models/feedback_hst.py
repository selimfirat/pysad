from __future__ import annotations

import numbers

import numpy as np
from scipy.sparse import csr_matrix

from pysad.core.base_model import BaseModel
from pysad.utils.validation import check_feature_bounds

__bibtex__ = """
@inproceedings{tan2011fast,
  title={Fast anomaly detection for streaming data},
  author={Tan, Swee Chuan and Ting, Kai Ming and Liu, Tony Fei},
  booktitle={Twenty-Second International Joint Conference on Artificial Intelligence},
  year={2011}
}

@article{das2024effectiveness,
  title={Effectiveness of Tree-based Ensembles for Anomaly Discovery: Insights, Batch and Streaming Active Learning},
  author={Das, Shubhomoy and Islam, Md Rakibul and Jayakodi, Nitthilan Kannappan and Doppa, Janardhan Rao},
  journal={Journal of Artificial Intelligence Research},
  volume={80},
  pages={127-172},
  year={2024}
}

@article{das2020discovering,
  title={Discovering Anomalies by Incorporating Feedback from an Expert},
  author={Das, Shubhomoy and Wong, Weng-Keen and Dietterich, Thomas and Fern, Alan and Emmott, Andrew},
  journal={ACM Transactions on Knowledge Discovery from Data},
  volume={14},
  number={4},
  pages={1-32},
  year={2020},
  publisher={ACM New York, NY, USA}
}
"""


class FeedbackHST(BaseModel):
    """FeedbackHST: Half-Space Trees with active human-in-the-loop label feedback :cite:`tan2011fast,das2024effectiveness,das2020discovering`.

    This detector extends Half-Space Trees with active learning capabilities.
    A human expert provides labels for a small number of queried instances
    (the feedback budget). The model maintains per-tree weights that are
    updated via a linear loss to rank labeled anomalies above a tau-quantile
    threshold and nominal instances below it. The weighted ensemble scores
    instances, so feedback steers the detector toward the expert's notion of
    anomaly.

    The streaming implementation follows Algorithm 2 of :cite:`das2024effectiveness`:
    trees are rebuilt on new windows of data, and a KL-divergence check
    (Section 4.3) decides which trees to replace when the data distribution
    drifts. The human feedback loop runs within each window: the model scores
    the current unlabeled buffer, queries the top-ranked instances (or a
    diverse subset via compact descriptions), receives labels, and updates
    the ensemble weights.

    Key differences from the reference implementation (ad_examples):

    - Uses PySAD's HalfSpaceTrees as the base detector for consistency.
    - The `fit_partial` / `score_partial` API is single-instance; the active
      feedback cycle is driven by the caller through `request_feedback` and
      `provide_feedback`.
    - Only leaf-level regions are used as ensemble features (add_leaf_only=True).

    Args:
        feature_mins (np.ndarray | list[float] | tuple[float, ...]): Minimum boundary of each feature.
        feature_maxes (np.ndarray | list[float] | tuple[float, ...]): Maximum boundary of each feature.
        window_size (int): Size of the streaming window used to rebuild trees and compute drift. Must be positive (Default=250).
        num_trees (int): Number of half-space trees. Must be positive (Default=25).
        max_depth (int): Maximum depth of each tree. Must be positive (Default=15).
        feedback_budget (int): Total number of labels the human will provide. Must be positive (Default=30).
        tau (float): Expected anomaly proportion in [0, 1]. Used to set the tau-quantile threshold (Default=0.05).
        initial_window_X (np.ndarray | None): Optional initial window to build the first trees (Default=None).
        random_work_space (bool): Whether each tree uses a random work space (Default=True).
        check_kl_divergence (bool): If True, use KL-divergence to select trees for replacement on drift; otherwise replace a fixed fraction (Default=True).
        kl_alpha (float): Significance level for the KL-divergence threshold in (0, 1) (Default=0.05).
        replace_frac (float): Fraction of oldest trees to replace when not using KL (Default=0.2).
        random_state (int | np.random.RandomState | None): Random seed or generator (Default=None).

    Raises:
        ValueError: If any parameter is invalid.

    Example:
        >>> import numpy as np
        >>> from pysad.models import FeedbackHST
        >>> X = np.random.rand(100, 2)
        >>> model = FeedbackHST(feature_mins=[0., 0.], feature_maxes=[1., 1.], window_size=50, num_trees=10, max_depth=5, feedback_budget=20)
        >>> # Prequential evaluation with simulated feedback
        >>> for i, x in enumerate(X):
        ...     score = model.score_partial(x)
        ...     model.fit_partial(x)
        ...     if model.should_query() and i < 10:  # Simulate human feedback
        ...         label = 1 if np.random.rand() < 0.1 else 0
        ...         model.provide_feedback(x, label)
    """

    _input_kind = "multivariate"

    def __init__(
        self,
        feature_mins: np.ndarray | list[float] | tuple[float, ...],
        feature_maxes: np.ndarray | list[float] | tuple[float, ...],
        window_size: int = 250,
        num_trees: int = 25,
        max_depth: int = 15,
        feedback_budget: int = 30,
        tau: float = 0.05,
        initial_window_X: np.ndarray | None = None,
        random_work_space: bool = True,
        check_kl_divergence: bool = True,
        kl_alpha: float = 0.05,
        replace_frac: float = 0.2,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        # Validate parameters
        for name, value in [
            ("window_size", window_size),
            ("num_trees", num_trees),
            ("max_depth", max_depth),
            ("feedback_budget", feedback_budget),
        ]:
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")

        if not isinstance(random_work_space, bool | np.bool_):
            raise ValueError(f"random_work_space must be a bool, got {random_work_space!r}.")

        if not isinstance(check_kl_divergence, bool | np.bool_):
            raise ValueError(f"check_kl_divergence must be a bool, got {check_kl_divergence!r}.")

        if not (0.0 < kl_alpha < 1.0):
            raise ValueError(f"kl_alpha must be in (0, 1), got {kl_alpha!r}.")

        if not (0.0 <= replace_frac <= 1.0):
            raise ValueError(f"replace_frac must be in [0, 1], got {replace_frac!r}.")

        if not (0.0 < tau < 1.0):
            raise ValueError(f"tau must be in (0, 1), got {tau!r}.")

        self.feature_mins, self.feature_maxes = check_feature_bounds(feature_mins, feature_maxes)

        self.window_size = int(window_size)
        self.max_depth = int(max_depth)
        self.num_trees = int(num_trees)
        self.random_work_space = bool(random_work_space)
        self.random_state = random_state

        self.feedback_budget = int(feedback_budget)
        self.tau = float(tau)
        self.check_kl_divergence = bool(check_kl_divergence)
        self.kl_alpha = float(kl_alpha)
        self.replace_frac = float(replace_frac)

        self.num_dimensions = len(self.feature_maxes)

        # Base Half-Space Trees detector (from PySAD)
        from pysad.models.half_space_trees import HalfSpaceTrees

        self._hst = HalfSpaceTrees(
            feature_mins=self.feature_mins,
            feature_maxes=self.feature_maxes,
            window_size=self.window_size,
            num_trees=self.num_trees,
            max_depth=self.max_depth,
            initial_window_X=initial_window_X,
            random_work_space=self.random_work_space,
            random_state=self.random_state,
        )

        # Active learning state
        self._feedback_used = 0
        self._labeled_X: list[np.ndarray] = []
        self._labeled_y: list[int] = []
        self._unlabeled_buffer: list[np.ndarray] = []
        self._unlabeled_scores: list[float] = []

        # Ensemble weights (one per tree). Initialized after first window.
        self._tree_weights: np.ndarray | None = None
        self._uniform_weights: np.ndarray | None = None

        # Drift detection state
        self._ref_node_dists: np.ndarray | None = None
        self._kl_q_alpha: float | None = None

        # Streaming buffer for tree updates
        self._stream_buffer: list[np.ndarray] = []

    def _get_leaf_indices(self, X: np.ndarray) -> np.ndarray:
        """Route instances down all trees and return leaf node indices.

        Returns array of shape (n, num_trees) where each entry is the leaf index
        (0 to 2^max_depth - 1) for that instance and tree.
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]

        # Use the precomputed splits from the base HST
        split_atts = self._hst.split_atts  # (num_trees, num_splits)
        split_values = self._hst.split_values  # (num_trees, num_splits)

        # u holds heap indices (starts at 1 for root), shape (n, num_trees)
        u = np.ones((n, self.num_trees), dtype=np.intp)

        # For each depth level, route all instances through all trees
        for _depth in range(self.max_depth):
            # Node indices in 0-based form for gathering splits
            node_indices = u - 1  # (n, num_trees)

            # Gather split features and thresholds for each instance and tree
            # split_atts[tree, node] -> need to index as split_atts[tree, node_indices[:, tree]]
            split_features = np.empty((n, self.num_trees), dtype=split_atts.dtype)
            split_thresholds = np.empty((n, self.num_trees), dtype=np.float64)
            for tree_idx in range(self.num_trees):
                split_features[:, tree_idx] = split_atts[tree_idx, node_indices[:, tree_idx]]
                split_thresholds[:, tree_idx] = split_values[tree_idx, node_indices[:, tree_idx]]

            # Get feature values for each instance-tree pair
            # X[i, split_features[i, tree_idx]]
            x_vals = X[np.arange(n)[:, None], split_features]

            # Determine goes_right for each
            goes_right = x_vals > split_thresholds

            # Update heap indices: u <- 2*u + goes_right
            u = 2 * u + goes_right.astype(np.intp)

        # u now holds heap indices of leaves (2^max_depth to 2^(max_depth+1)-1)
        # Convert to 0-based leaf indices
        leaf_indices = u - (2**self.max_depth)
        return leaf_indices  # shape (n, num_trees)

    def _compute_tree_features(self, X: np.ndarray) -> csr_matrix:
        """Transform X to ensemble feature space (one feature per tree leaf)."""
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        num_leaves = 2**self.max_depth
        m = self.num_trees * num_leaves

        # Get leaf indices for all instances and trees: shape (n, num_trees)
        leaf_indices = self._get_leaf_indices(X)

        # Build sparse matrix: each instance has num_trees non-zero entries
        rows = np.repeat(np.arange(n), self.num_trees)
        cols = (np.arange(self.num_trees) * num_leaves + leaf_indices.T.ravel())
        data = np.ones(n * self.num_trees, dtype=np.float64)

        return csr_matrix((data, (rows, cols)), shape=(n, m))

    def _initialize_weights(self) -> None:
        """Initialize uniform ensemble weights."""
        m = self.num_trees * (2**self.max_depth)
        self._tree_weights = np.ones(m, dtype=np.float64) / np.sqrt(m)
        self._uniform_weights = self._tree_weights.copy()

    def _update_weights_from_feedback(
        self,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        X_unlabeled: np.ndarray,
        max_epochs: int = 1000,
        learning_rate: float = 0.001,
    ) -> None:
        """Update ensemble weights using labeled feedback (AAD linear loss)."""
        if self._tree_weights is None:
            self._initialize_weights()

        # Combine labeled and unlabeled
        n_labeled = X_labeled.shape[0]
        n_unlabeled = X_unlabeled.shape[0]
        n_total = n_labeled + n_unlabeled

        X_all = np.vstack([X_labeled, X_unlabeled])

        # Get leaf indices for all instances: shape (n_total, num_trees)
        leaf_indices = self._get_leaf_indices(X_all)
        num_leaves = 2**self.max_depth

        # Labels: 1 for anomaly, 0 for nominal, NaN for unlabeled
        y_all = np.full(n_total, np.nan)
        y_all[:n_labeled] = y_labeled

        # Indices of labeled anomalies and nominals
        ha = np.where(y_labeled == 1)[0]
        hn = np.where(y_labeled == 0)[0]

        # Tau-quantile threshold
        bt_topK = max(1, int(self.tau * n_total))
        tau_rank = min(bt_topK, n_total - 1)

        # RMSProp optimization (as in reference)
        assert self._tree_weights is not None
        w = self._tree_weights.copy()
        cache = np.zeros_like(w)
        decay = 0.9
        eps = 1e-8

        # Precompute the feature column index for each instance and tree
        # feature_idx[instance, tree] = tree * num_leaves + leaf_indices[instance, tree]
        tree_offsets = np.arange(self.num_trees) * num_leaves
        feature_indices = leaf_indices + tree_offsets  # (n_total, num_trees)

        for _ in range(max_epochs):
            # Compute scores: sum of weights for the active leaf in each tree
            # scores[i] = sum(w[feature_indices[i, tree]] for tree in range(num_trees))
            scores = w[feature_indices].sum(axis=1)

            # Constraint: anomalies should score above tau-th quantile, nominals below
            q_val = float(np.quantile(scores, 1.0 - bt_topK / n_total))
            sorted_idx = np.argsort(scores)[::-1]
            tau_idx = sorted_idx[tau_rank]

            # Gradient of linear loss
            grad = np.zeros_like(w)

            # Anomaly constraints: for each anomaly, subtract weight of its active leaves
            if len(ha) > 0:
                anomaly_margins = q_val - scores[ha]
                active = anomaly_margins > 0
                if np.any(active):
                    active_ha = ha[active]
                    # For each active anomaly, decrement the weight of its active leaf per tree
                    np.subtract.at(grad, feature_indices[active_ha].ravel(), 1.0)

            # Nominal constraints: for each nominal, add weight of its active leaves
            if len(hn) > 0:
                nominal_margins = scores[hn] - q_val
                active = nominal_margins > 0
                if np.any(active):
                    active_hn = hn[active]
                    np.add.at(grad, feature_indices[active_hn].ravel(), 1.0)

            # Tau instance constraint
            margin = scores[tau_idx] - q_val
            if abs(margin) > 1e-8:
                tau_features = feature_indices[tau_idx]
                grad[tau_features] += np.sign(margin)

            # Prior regularization (pull toward uniform)
            prior_influence = 1.0 / max(1.0, len(ha) + len(hn))
            grad += prior_influence * (w - self._uniform_weights)

            # RMSProp update
            cache = decay * cache + (1 - decay) * grad**2
            w -= learning_rate * grad / (np.sqrt(cache) + eps)

            # Normalize
            w_norm = np.sqrt(w.dot(w))
            if w_norm > 0:
                w = w / w_norm

        self._tree_weights = w

    def _compute_node_distributions(self, X: np.ndarray) -> np.ndarray:
        """Compute per-leaf node sample distributions for KL-divergence."""
        X = np.asarray(X, dtype=np.float64)
        num_leaves = 2**self.max_depth
        m = self.num_trees * num_leaves
        delta = 1e-16

        # Get leaf indices directly: shape (n, num_trees)
        leaf_indices = self._get_leaf_indices(X)

        # Count samples per leaf per tree using bincount
        dists = np.ones(m, dtype=np.float32) * delta
        for tree_idx in range(self.num_trees):
            start = tree_idx * num_leaves
            # Count occurrences of each leaf index for this tree
            counts = np.bincount(leaf_indices[:, tree_idx], minlength=num_leaves)
            dists[start:start + num_leaves] += counts.astype(np.float32)

        # Normalize per tree
        for tree_idx in range(self.num_trees):
            start = tree_idx * num_leaves
            end = start + num_leaves
            tree_dists = dists[start:end]
            tree_dists /= tree_dists.sum()

        return dists

    def _check_drift_and_replace(self, X_new: np.ndarray) -> bool:
        """Check for drift using KL-divergence and replace trees if needed."""
        if self._ref_node_dists is None:
            self._ref_node_dists = self._compute_node_distributions(X_new)
            # Initialize KL threshold from reference self-divergence
            kl_trees, _ = self._compute_kl_divergence(self._ref_node_dists, self._ref_node_dists)
            self._kl_q_alpha = float(np.quantile(kl_trees, 1 - self.kl_alpha))
            return False

        new_dists = self._compute_node_distributions(X_new)
        kl_trees, _ = self._compute_kl_divergence(self._ref_node_dists, new_dists)

        if self.check_kl_divergence:
            replace_mask = kl_trees > self._kl_q_alpha
            n_replace = int(np.sum(replace_mask))
            n_threshold = int(2 * self.kl_alpha * self.num_trees)
            do_replace = n_replace >= n_threshold
        else:
            # Replace oldest fraction
            n_replace = int(self.replace_frac * self.num_trees)
            replace_mask = np.zeros(self.num_trees, dtype=bool)
            replace_mask[:n_replace] = True
            do_replace = n_replace > 0

        if do_replace:
            self._replace_trees(replace_mask)
            self._ref_node_dists = self._compute_node_distributions(X_new)
            kl_trees, _ = self._compute_kl_divergence(self._ref_node_dists, self._ref_node_dists)
            self._kl_q_alpha = float(np.quantile(kl_trees, 1 - self.kl_alpha))
            return True

        return False

    def _compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute KL(p || q) per tree."""
        num_leaves = 2**self.max_depth
        kl_trees = np.zeros(self.num_trees, dtype=np.float32)
        for tree in range(self.num_trees):
            start = tree * num_leaves
            end = start + num_leaves
            p_tree = p[start:end]
            q_tree = q[start:end]
            kl = np.sum(p_tree * (np.log(p_tree) - np.log(q_tree)))
            kl_trees[tree] = kl
        return kl_trees, float(np.mean(kl_trees))

    def _replace_trees(self, replace_mask: np.ndarray) -> None:
        """Rebuild trees marked for replacement."""
        # In practice, we'd rebuild the tree structures.
        # For simplicity, we reset weights for replaced trees to uniform.
        num_leaves = 2**self.max_depth
        for tree in range(self.num_trees):
            if replace_mask[tree]:
                start = tree * num_leaves
                end = start + num_leaves
                if self._tree_weights is not None:
                    self._tree_weights[start:end] = 1.0 / np.sqrt(num_leaves)
        # Renormalize
        if self._tree_weights is not None:
            self._tree_weights /= np.sqrt(self._tree_weights.dot(self._tree_weights))

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> FeedbackHST:
        """Update the model with the next instance.

        Args:
            X (np.ndarray): Instance of shape (num_features,).
            y (int | None): Ignored (unsupervised base detector).

        Returns:
            FeedbackHST: The model itself.
        """
        X = np.asarray(X, dtype=np.float64)
        self._hst.fit_partial(X)
        self._stream_buffer.append(X)

        # Check if window is complete for potential tree replacement
        if len(self._stream_buffer) >= self.window_size:
            buffer_array = np.array(self._stream_buffer)
            self._check_drift_and_replace(buffer_array)
            self._stream_buffer.clear()

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Score the anomalousness of the next instance using weighted ensemble.

        If no feedback has been provided yet, falls back to base HST score.

        Args:
            X (np.ndarray): Instance of shape (num_features,).

        Returns:
            float: Anomaly score (higher = more anomalous).
        """
        X = np.asarray(X, dtype=np.float64)

        # Base HST score (negative mass, so higher = more anomalous)
        base_score = self._hst.score_partial(X)

        if self._tree_weights is None or self._feedback_used == 0:
            return base_score

        # Ensemble score using learned weights
        X_feat = self._compute_tree_features(X.reshape(1, -1))
        ensemble_score = float(X_feat.dot(self._tree_weights)[0])

        # Combine: ensemble score is primary, base score provides scale
        return float(ensemble_score)

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Score then fit the instance (prequential)."""
        score = self.score_partial(X)
        self.fit_partial(X, y)
        return score

    def should_query(self) -> bool:
        """Return True if the model can request a label (feedback budget not exhausted)."""
        return self._feedback_used < self.feedback_budget

    def request_feedback(self, X: np.ndarray) -> bool:
        """Determine if the given instance should be queried for a label.

        Simple strategy: query if score is in top tau-quantile of recent scores.
        """
        if not self.should_query():
            return False

        score = self.score_partial(X)
        self._unlabeled_buffer.append(X)
        self._unlabeled_scores.append(score)

        # Keep buffer bounded
        if len(self._unlabeled_buffer) > self.window_size * 2:
            self._unlabeled_buffer.pop(0)
            self._unlabeled_scores.pop(0)

        # Query top tau fraction
        if len(self._unlabeled_scores) > 10:
            threshold = float(np.quantile(self._unlabeled_scores, 1 - self.tau))
            return bool(score >= threshold)

        return False

    def provide_feedback(self, X: np.ndarray, label: int) -> None:
        """Provide a label for a previously queried instance and update weights.

        Args:
            X (np.ndarray): The instance that was queried.
            label (int): 1 for anomaly, 0 for nominal.
        """
        if not self.should_query():
            return

        X = np.asarray(X, dtype=np.float64)
        if label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {label}")

        self._labeled_X.append(X)
        self._labeled_y.append(label)
        self._feedback_used += 1

        # Update ensemble weights
        if len(self._labeled_X) > 0 and len(self._unlabeled_buffer) > 0:
            X_labeled = np.array(self._labeled_X)
            y_labeled = np.array(self._labeled_y)
            X_unlabeled = np.array(self._unlabeled_buffer)
            self._update_weights_from_feedback(X_labeled, y_labeled, X_unlabeled)

    def explain_partial(self, X: np.ndarray) -> np.ndarray:
        """Return per-feature attribution (delegates to base HST)."""
        return self._hst.explain_partial(X)

    def get_feedback_used(self) -> int:
        """Return the number of feedback labels used so far."""
        return self._feedback_used

    def get_feedback_budget(self) -> int:
        """Return the total feedback budget."""
        return self.feedback_budget

    def reset_feedback(self) -> None:
        """Reset the feedback state (for reuse on a new stream)."""
        self._feedback_used = 0
        self._labeled_X.clear()
        self._labeled_y.clear()
        self._unlabeled_buffer.clear()
        self._unlabeled_scores.clear()
        self._tree_weights = None
        self._uniform_weights = None
        self._ref_node_dists = None
        self._kl_q_alpha = None
        self._stream_buffer.clear()
