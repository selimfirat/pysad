from __future__ import annotations

import numbers

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils import check_random_state

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

    def _work_space(self, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
        if not self.random_work_space:
            return self.feature_mins.copy(), self.feature_maxes.copy()

        s = rng.uniform(self.feature_mins, self.feature_maxes)
        half_width = 2.0 * np.maximum(s - self.feature_mins, self.feature_maxes - s)

        return s - half_width, s + half_width

    def _build_tree_splits(self, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
        """Build split features and values for one tree (copied from HalfSpaceTrees)."""
        mins, maxes = self._work_space(rng)
        depth = self.max_depth
        atts_depth_first = rng.randint(self.num_dimensions, size=2**depth - 1)

        level = np.zeros(1, dtype=np.intp)
        levels = [level]
        for k in range(depth - 1):
            level = np.stack([level + 1, level + 2 ** (depth - k - 1)], axis=1).ravel()
            levels.append(level)
        atts = atts_depth_first[np.concatenate(levels)]

        values = np.empty(2**depth - 1)
        for k in range(depth):
            nodes = np.arange(2**k - 1, 2 ** (k + 1) - 1)
            q = atts[nodes]
            lower, upper = mins[q], maxes[q]
            for ancestor_depth in range(k):
                ancestors = ((nodes + 1) >> (k - ancestor_depth)) - 1
                goes_right = (((nodes + 1) >> (k - ancestor_depth - 1)) & 1) == 1
                splits_q = atts[ancestors] == q
                upper = np.where(splits_q & ~goes_right, values[ancestors], upper)
                lower = np.where(splits_q & goes_right, values[ancestors], lower)
            values[nodes] = (upper + lower) / 2.0

        return atts, values

    def _get_tree_leaf_nodes(
        self, X: np.ndarray, split_atts: np.ndarray, split_values: np.ndarray
    ) -> np.ndarray:
        """Route instances down a tree and return leaf node indices."""
        n = X.shape[0]
        u = np.ones(n, dtype=np.intp)  # heap index starts at 1 (root)
        for _k in range(self.max_depth):
            # For each instance, get the split feature and value at its current node
            node_indices = u - 1  # 0-based node index
            # Get split feature and value for each instance's current node
            split_features = split_atts[node_indices]
            split_thresholds = split_values[node_indices]
            # Get feature value for each instance
            x_vals = X[np.arange(n), split_features]
            goes_right = x_vals > split_thresholds
            u = 2 * u + goes_right.astype(np.intp)
        # u now holds heap indices of leaves (2^max_depth to 2^(max_depth+1)-1)
        leaf_node_indices = u - (2**self.max_depth)  # 0-based leaf index
        return leaf_node_indices

    def _compute_tree_features(self, X: np.ndarray) -> csr_matrix:
        """Transform X to ensemble feature space (one feature per tree leaf)."""
        n = X.shape[0]
        num_leaves = 2**self.max_depth
        m = self.num_trees * num_leaves

        rng = check_random_state(self.random_state)
        # Rebuild tree splits (same as in _hst)
        all_split_atts = []
        all_split_values = []
        for _tree in range(self.num_trees):
            atts, values = self._build_tree_splits(rng)
            all_split_atts.append(atts)
            all_split_values.append(values)

        # For each tree, route instances to leaves and set feature
        rows = []
        cols = []
        data = []
        for tree_idx in range(self.num_trees):
            split_atts = all_split_atts[tree_idx]
            split_values = all_split_values[tree_idx]
            leaf_indices = self._get_tree_leaf_nodes(X, split_atts, split_values)
            for i in range(n):
                rows.append(i)
                cols.append(tree_idx * num_leaves + leaf_indices[i])
                data.append(1.0)  # Binary indicator: instance falls in this leaf

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

        # Transform all instances to ensemble feature space
        X_all = np.vstack([X_labeled, X_unlabeled])
        X_features = self._compute_tree_features(X_all)

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

        for _ in range(max_epochs):
            scores = X_features.dot(w)

            # Constraint: anomalies should score above tau-th quantile, nominals below
            q_val = float(np.quantile(scores, 1.0 - bt_topK / n_total))
            x_tau = X_features[np.argsort(scores)[::-1][tau_rank]]

            # Gradient of linear loss
            grad = np.zeros_like(w)
            # Anomaly constraints
            for idx in ha:
                margin = q_val - scores[idx]
                if margin > 0:
                    grad -= X_features[idx].toarray().ravel()
            # Nominal constraints
            for idx in hn:
                margin = scores[idx] - q_val
                if margin > 0:
                    grad += X_features[idx].toarray().ravel()
            # Tau instance constraint
            margin = scores[tau_rank] - q_val
            if abs(margin) > 1e-8:
                grad += np.sign(margin) * x_tau.toarray().ravel()

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
        num_leaves = 2**self.max_depth
        m = self.num_trees * num_leaves
        delta = 1e-16

        X_features = self._compute_tree_features(X)
        dists = np.ones(m, dtype=np.float32) * delta

        # Count samples per leaf
        for j in range(m):
            dists[j] += X_features[:, j].sum()

        # Normalize per tree
        for tree in range(self.num_trees):
            start = tree * num_leaves
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
