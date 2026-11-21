from __future__ import annotations

import numbers
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.models.half_space_trees import HalfSpaceTrees
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
    (the feedback budget). The model maintains per-region weights that are
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
    - Query diversity via compact descriptions (QUERY_TYPE=8 in reference) is
      not implemented; only the top-tau strategy (QUERY_TYPE=1) is supported.
      This is a deliberate simplification for the initial PySAD integration.
    - Tree replacement on drift rebuilds the half-space splits for selected
      trees using new window data. The reference implementation only updates
      node counts and raises NotImplementedError for HSTree replacement. This
      deviation enables full tree adaptation to concept drift in the streaming
      setting.
    - **KL-divergence threshold**: The reference computes a per-tree KL
      distribution from repeated random half-splits of the reference window
      and thresholds each tree individually. This implementation builds the
      same per-tree distribution but uses the reference window data directly
      rather than sampling from feature bounds. The threshold logic is
      otherwise aligned (per-tree quantile at 1 - kl_alpha).
    - **Weight update loss**: The reference uses a hinge loss with class-weight
      parameters Ca, Cn, Cx, class-count normalization, a tau-relative term,
      and an L2 prior with sigma2 scaling, optimized via gradient descent.
      This implementation uses a simplified hinge loss without the Ca/Cn/Cx
      parameters or class-count normalization, omits the tau-relative term,
      scales the L2 prior by `prior_influence / (num_labeled_anomalies + num_labeled_nominals)`,
      and uses RMSProp instead of gradient descent. Hyperparameters
      (`learning_rate`, `max_epochs`, `rmsprop_decay`, `rmsprop_eps`,
      `prior_influence`) are configurable.
    - **Score transition window**: A `score_transition_window` parameter
      (default=50) blends base HST and ensemble scores over the first N
      instances after feedback starts to avoid a score discontinuity. This
      parameter and behavior are not in the paper or reference implementation
      and have no citation. Set to 0 for immediate switch to ensemble scores.
    - **Reference region estimation for KL check**: The reference uses the
      actual reference window distributions. This implementation also uses
      the reference window data (split into random halves) rather than
      sampling random points from feature bounds.
    - **Weight initialization after tree replacement**: Replaced trees reset
      their region weights to uniform values (1/sqrt(num_leaves)) and the
      full weight vector is renormalized. The retained trees' weights are
      kept but participate in the renormalization. The reference does not
      implement HSTree replacement, so no direct comparison exists.

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
        learning_rate (float): Learning rate for RMSProp weight optimization (Default=0.001).
        max_epochs (int): Maximum optimization epochs per weight update (Default=1000).
        rmsprop_decay (float): RMSProp decay factor in [0, 1) (Default=0.9).
        rmsprop_eps (float): RMSProp epsilon for numerical stability (Default=1e-8).
        prior_influence (float): Prior regularization strength pulling weights toward uniform (Default=1.0).
        score_transition_window (int): Number of instances over which to blend base HST
            and ensemble scores after first feedback (Default=50). Set to 0 for immediate switch.

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

    # Configuration parameters
    feature_mins: NDArray[np.float64]
    feature_maxes: NDArray[np.float64]
    window_size: int
    max_depth: int
    num_trees: int
    random_work_space: bool
    random_state: int | np.random.RandomState | None
    feedback_budget: int
    tau: float
    check_kl_divergence: bool
    kl_alpha: float
    replace_frac: float
    learning_rate: float
    max_epochs: int
    rmsprop_decay: float
    rmsprop_eps: float
    prior_influence: float
    score_transition_window: int
    num_dimensions: int

    # Base detector
    _hst: HalfSpaceTrees
    _rng: np.random.RandomState

    # Active learning state
    _feedback_used: int
    _labeled_X: list[NDArray[np.float64]]
    _labeled_y: list[int]
    _unlabeled_buffer: list[NDArray[np.float64]]
    _unlabeled_scores: list[float]
    _feedback_started: bool
    _instances_since_feedback: int

    # Ensemble weights
    _region_weights: NDArray[np.float64] | None
    _uniform_weights: NDArray[np.float64] | None
    _region_scores: NDArray[np.float64] | None

    # Drift detection state
    _ref_node_dists: NDArray[np.float32] | None
    _kl_q_alpha: NDArray[np.float32] | None
    _drift_ref_initialized: bool

    # Streaming buffer
    _stream_buffer: list[NDArray[np.float64]]

    def __init__(
        self,
        feature_mins: NDArray[np.float64] | list[float] | tuple[float, ...],
        feature_maxes: NDArray[np.float64] | list[float] | tuple[float, ...],
        window_size: int = 250,
        num_trees: int = 25,
        max_depth: int = 15,
        feedback_budget: int = 30,
        tau: float = 0.05,
        initial_window_X: NDArray[np.float64] | None = None,
        random_work_space: bool = True,
        check_kl_divergence: bool = True,
        kl_alpha: float = 0.05,
        replace_frac: float = 0.2,
        random_state: int | np.random.RandomState | None = None,
        # Loss optimization hyperparameters (match reference implementation defaults)
        learning_rate: float = 0.001,
        max_epochs: int = 1000,
        rmsprop_decay: float = 0.9,
        rmsprop_eps: float = 1e-8,
        prior_influence: float = 1.0,
        # Score transition: number of instances to blend base HST and ensemble scores
        # after first feedback to avoid discontinuity. 0 = immediate switch.
        score_transition_window: int = 50,
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

        if not (0.0 < learning_rate):
            raise ValueError(f"learning_rate must be positive, got {learning_rate!r}.")

        if not isinstance(max_epochs, numbers.Integral) or max_epochs < 1:
            raise ValueError(f"max_epochs must be a positive integer, got {max_epochs!r}.")

        if not (0.0 <= rmsprop_decay < 1.0):
            raise ValueError(f"rmsprop_decay must be in [0, 1), got {rmsprop_decay!r}.")

        if not (0.0 < rmsprop_eps):
            raise ValueError(f"rmsprop_eps must be positive, got {rmsprop_eps!r}.")

        if not (0.0 <= prior_influence):
            raise ValueError(f"prior_influence must be non-negative, got {prior_influence!r}.")

        if not isinstance(score_transition_window, numbers.Integral) or score_transition_window < 0:
            raise ValueError(
                f"score_transition_window must be a non-negative integer, got {score_transition_window!r}."
            )

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

        # Loss optimization hyperparameters
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.rmsprop_decay = float(rmsprop_decay)
        self.rmsprop_eps = float(rmsprop_eps)
        self.prior_influence = float(prior_influence)
        self.score_transition_window = int(score_transition_window)

        self.num_dimensions = len(self.feature_maxes)

        # Random state for reproducible randomness
        if random_state is None:
            # Create an independent RandomState initialized from the global state
            global_rng = check_random_state(None)
            self._rng = np.random.RandomState()
            self._rng.set_state(global_rng.get_state())
        else:
            self._rng = check_random_state(random_state)

        # Base Half-Space Trees detector (from PySAD) - pass our RandomState for consistency
        self._hst = HalfSpaceTrees(
            feature_mins=self.feature_mins,
            feature_maxes=self.feature_maxes,
            window_size=self.window_size,
            num_trees=self.num_trees,
            max_depth=self.max_depth,
            initial_window_X=initial_window_X,
            random_work_space=self.random_work_space,
            random_state=self._rng,
        )

        # Active learning state
        self._feedback_used = 0
        self._labeled_X: list[NDArray[np.float64]] = []
        self._labeled_y: list[int] = []
        self._unlabeled_buffer: list[NDArray[np.float64]] = []
        self._unlabeled_scores: list[float] = []

        # Ensemble weights (one per tree leaf region). Initialized after first window.
        self._region_weights: NDArray[np.float64] | None = None
        self._uniform_weights: NDArray[np.float64] | None = None

        # Region scores (d) for each leaf region: -frac_insts * 2^path_length
        self._region_scores: NDArray[np.float64] | None = None

        # Drift detection state
        self._ref_node_dists: NDArray[np.float32] | None = None
        self._kl_q_alpha: NDArray[np.float32] | None = None

        # Score transition state
        self._feedback_started = False
        self._instances_since_feedback = 0

        # Streaming buffer for tree updates
        self._stream_buffer: list[NDArray[np.float64]] = []

        # Track if we've initialized drift reference from initial window
        self._drift_ref_initialized = False

    def _get_leaf_indices(self, X: NDArray[np.float64]) -> NDArray[np.intp]:
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
            split_features = np.empty((n, self.num_trees), dtype=split_atts.dtype)
            split_thresholds = np.empty((n, self.num_trees), dtype=np.float64)
            for tree_idx in range(self.num_trees):
                split_features[:, tree_idx] = split_atts[tree_idx, node_indices[:, tree_idx]]
                split_thresholds[:, tree_idx] = split_values[tree_idx, node_indices[:, tree_idx]]

            # Get feature values for each instance-tree pair
            x_vals = X[np.arange(n)[:, None], split_features]

            # Determine goes_right for each
            goes_right = x_vals > split_thresholds

            # Update heap indices: u <- 2*u + goes_right
            u = 2 * u + goes_right.astype(np.intp)

        # u now holds heap indices of leaves (2^max_depth to 2^(max_depth+1)-1)
        # Convert to 0-based leaf indices
        leaf_indices: NDArray[np.intp] = u - (2**self.max_depth)
        return leaf_indices  # shape (n, num_trees)

    def _compute_region_scores(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute region scores for all leaf regions: -frac_insts * 2^path_length.

        This matches the HST_SCORE_TYPE from the reference implementation.
        Returns array of shape (num_trees * num_leaves,) where num_leaves = 2^max_depth.
        """
        num_leaves = 2**self.max_depth
        m = self.num_trees * num_leaves

        # Get leaf indices for the reference data
        leaf_indices = self._get_leaf_indices(X)
        n = leaf_indices.shape[0]

        # Count samples per leaf per tree
        region_scores = np.zeros(m, dtype=np.float64)
        for tree_idx in range(self.num_trees):
            start = tree_idx * num_leaves
            counts = np.bincount(leaf_indices[:, tree_idx], minlength=num_leaves)
            frac_insts = counts / max(1, n)
            path_length = self.max_depth
            region_scores[start : start + num_leaves] = -frac_insts * (2**path_length)

        return region_scores

    def _initialize_weights(self) -> None:
        """Initialize uniform ensemble weights."""
        m = self.num_trees * (2**self.max_depth)
        self._region_weights = np.ones(m, dtype=np.float64) / np.sqrt(m)
        self._uniform_weights = self._region_weights.copy()

    def _transform_to_ensemble_features(self, X: NDArray[np.float64]) -> csr_matrix:
        """Transform X to ensemble feature space using region scores.

        Each instance activates exactly one leaf per tree, with the feature value
        being the region score for that leaf. This matches the reference
        implementation's transform_to_ensemble_features with add_leaf_nodes_only=True.
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        num_leaves = 2**self.max_depth
        m = self.num_trees * num_leaves

        # Get leaf indices for all instances and trees: shape (n, num_trees)
        leaf_indices = self._get_leaf_indices(X)

        # Build sparse matrix: each instance has num_trees non-zero entries
        # Value is the region score for that leaf
        rows = np.repeat(np.arange(n), self.num_trees)
        cols = np.arange(self.num_trees) * num_leaves + leaf_indices.T.ravel()

        # Get region scores for each active leaf
        if self._region_scores is None:
            self._region_scores = self._compute_region_scores(X)

        data = self._region_scores[cols]

        return csr_matrix((data, (rows, cols)), shape=(n, m))

    def _update_weights_from_feedback(
        self,
        X_labeled: NDArray[np.float64],
        y_labeled: NDArray[Any],
        X_unlabeled: NDArray[np.float64],
    ) -> None:
        """Update ensemble weights using labeled feedback (AAD linear loss).

        This follows the reference implementation's aad_loss_gradient_linear
        with RMSProp optimization. Hyperparameters are configurable via the
        constructor (learning_rate, max_epochs, rmsprop_decay, rmsprop_eps,
        prior_influence).
        """
        if self._region_weights is None:
            self._initialize_weights()

        # Combine labeled and unlabeled
        n_labeled = X_labeled.shape[0]
        n_unlabeled = X_unlabeled.shape[0]
        n_total = n_labeled + n_unlabeled

        X_all = np.vstack([X_labeled, X_unlabeled])

        # Transform to ensemble feature space (sparse)
        X_feat = self._transform_to_ensemble_features(X_all)

        # Indices of labeled anomalies and nominals
        ha = np.where(y_labeled == 1)[0]
        hn = np.where(y_labeled == 0)[0]

        # Tau-quantile threshold
        bt_topK = max(1, int(self.tau * n_total))
        tau_rank = min(bt_topK, n_total - 1)

        # RMSProp optimization (as in reference)
        assert self._region_weights is not None
        assert self._uniform_weights is not None
        w = self._region_weights.copy()
        cache = np.zeros_like(w)
        decay = self.rmsprop_decay
        eps = self.rmsprop_eps

        for _ in range(self.max_epochs):
            # Compute scores: X_feat @ w
            scores = X_feat.dot(w).ravel()

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
                    X_feat_anom = X_feat[active_ha]
                    grad -= X_feat_anom.sum(axis=0).A1 / max(1, len(active_ha))

            # Nominal constraints: for each nominal, add weight of its active leaves
            if len(hn) > 0:
                nominal_margins = scores[hn] - q_val
                active = nominal_margins > 0
                if np.any(active):
                    active_hn = hn[active]
                    X_feat_noml = X_feat[active_hn]
                    grad += X_feat_noml.sum(axis=0).A1 / max(1, len(active_hn))

            # Tau instance constraint
            margin = scores[tau_idx] - q_val
            if abs(margin) > 1e-8:
                X_feat_tau = X_feat[tau_idx]
                grad += np.sign(margin) * X_feat_tau.A1

            # Prior regularization (pull toward uniform)
            prior_influence = self.prior_influence / max(1.0, len(ha) + len(hn))
            grad += prior_influence * (w - self._uniform_weights)

            # RMSProp update
            cache = decay * cache + (1 - decay) * grad**2
            w -= self.learning_rate * grad / (np.sqrt(cache) + eps)

            # Normalize
            w_norm = np.sqrt(w.dot(w))
            if w_norm > 0:
                w = w / w_norm

        self._region_weights = w

    def _compute_node_distributions(self, X: NDArray[np.float64]) -> NDArray[np.float32]:
        """Compute per-leaf node sample distributions for KL-divergence.

        Returns normalized distribution per tree (probabilities sum to 1 per tree).
        """
        X = np.asarray(X, dtype=np.float64)
        num_leaves = 2**self.max_depth
        m = self.num_trees * num_leaves
        delta = 1e-16

        # Get leaf indices directly: shape (n, num_trees)
        leaf_indices = self._get_leaf_indices(X)

        # Count samples per leaf per tree using bincount
        dists = np.full(m, delta, dtype=np.float32)
        for tree_idx in range(self.num_trees):
            start = tree_idx * num_leaves
            counts = np.bincount(leaf_indices[:, tree_idx], minlength=num_leaves)
            dists[start : start + num_leaves] += counts.astype(np.float32)

        # Normalize per tree
        for tree_idx in range(self.num_trees):
            start = tree_idx * num_leaves
            end = start + num_leaves
            tree_dists = dists[start:end]
            tree_dists /= tree_dists.sum()

        return dists

    def _initialize_drift_reference(self, X: NDArray[np.float64], n_splits: int = 30) -> None:
        """Initialize KL-divergence reference distributions by repeated random splits.

        This matches the reference implementation's get_KL_divergence_distribution
        with simple=True, which builds a distribution of KL values from repeated
        random half-splits of the reference data, then takes the (1-kl_alpha) quantile
        as the per-tree drift threshold.
        """
        self._ref_node_dists = self._compute_node_distributions(X)

        # Build KL distribution per tree via multiple random half-splits
        n = X.shape[0]
        kl_samples = np.zeros((n_splits, self.num_trees), dtype=np.float32)

        if n >= 2:
            for s in range(n_splits):
                indices = np.arange(n)
                self._rng.shuffle(indices)
                h = n // 2
                X1 = X[indices[:h]]
                X2 = X[indices[h:]]
                p1 = self._compute_node_distributions(X1)
                p2 = self._compute_node_distributions(X2)
                kl_trees, _ = self._compute_kl_divergence(p1, p2)
                kl_samples[s] = kl_trees
        else:
            # Not enough data to split, use self-divergence (will be 0)
            kl_trees, _ = self._compute_kl_divergence(self._ref_node_dists, self._ref_node_dists)
            kl_samples[:, :] = kl_trees

        # Set per-tree KL thresholds (quantile at 1 - kl_alpha) from the distribution
        self._kl_q_alpha = np.zeros(self.num_trees, dtype=np.float32)
        for i in range(self.num_trees):
            self._kl_q_alpha[i] = float(np.quantile(kl_samples[:, i], 1 - self.kl_alpha))

    def _check_drift_and_replace(self, X_new: NDArray[np.float64]) -> bool:
        """Check for drift using KL-divergence and replace trees if needed."""
        if not self._drift_ref_initialized:
            self._initialize_drift_reference(X_new)
            self._drift_ref_initialized = True
            return False

        new_dists = self._compute_node_distributions(X_new)
        assert self._ref_node_dists is not None
        kl_trees, _ = self._compute_kl_divergence(self._ref_node_dists, new_dists)

        if self.check_kl_divergence:
            assert self._kl_q_alpha is not None
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
            self._replace_trees(replace_mask, X_new)
            # Re-initialize KL threshold from new reference using multiple splits
            self._initialize_drift_reference(X_new)
            return True

        return False

    def _compute_kl_divergence(
        self, p: NDArray[np.float32], q: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float]:
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

    def _replace_trees(self, replace_mask: NDArray[np.bool_], X_new: NDArray[np.float64]) -> None:
        """Rebuild trees marked for replacement using new window data.

        This matches the reference implementation which constructs new trees
        from the new window data (Algorithm 2: "trees are rebuilt on new windows of data").
        """
        num_splits = 2**self.max_depth - 1
        num_leaves = 2**self.max_depth

        for tree in range(self.num_trees):
            if replace_mask[tree]:
                # Rebuild this tree's splits using the same logic as HalfSpaceTrees
                mins, maxes = self._hst._work_space(self._rng)
                atts_depth_first = self._rng.randint(self.num_dimensions, size=num_splits)

                # Position in depth-first order of every internal node, level by level in heap order
                level = np.zeros(1, dtype=np.intp)
                levels = [level]
                for k in range(self.max_depth - 1):
                    level = np.stack(
                        [level + 1, level + 2 ** (self.max_depth - k - 1)], axis=1
                    ).ravel()
                    levels.append(level)
                atts = atts_depth_first[np.concatenate(levels)]

                values = np.empty(num_splits)
                for k in range(self.max_depth):
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

                self._hst.split_atts[tree] = atts
                self._hst.split_values[tree] = values

                # Reset masses for this tree
                self._hst.l_mass[tree].fill(0)
                self._hst.r_mass[tree].fill(0)

                # Reset weights for replaced tree's regions to uniform
                if self._region_weights is not None:
                    start = tree * num_leaves
                    end = start + num_leaves
                    self._region_weights[start:end] = 1.0 / np.sqrt(num_leaves)

        # Recompute region scores from new window data
        self._region_scores = self._compute_region_scores(X_new)

        # Renormalize weights
        if self._region_weights is not None:
            self._region_weights /= np.sqrt(self._region_weights.dot(self._region_weights))
            self._uniform_weights = np.ones_like(self._region_weights) / np.sqrt(
                len(self._region_weights)
            )

        # Reset HST's window state so it starts fresh with new trees
        self._hst.is_first_window = True
        self._hst.step = 0

    def fit_partial(self, X: NDArray[np.float64], y: int | None = None) -> FeedbackHST:
        """Update the model with the next instance.

        Args:
            X (np.ndarray): Instance of shape (num_features,).
            y (int | None): Ignored (unsupervised base detector).

        Returns:
            FeedbackHST: The model itself.
        """
        X = np.asarray(X, dtype=np.float64)
        self._hst.fit_partial(X)
        self._stream_buffer.append(X.copy())

        # Check if window is complete for potential tree replacement
        if len(self._stream_buffer) >= self.window_size:
            buffer_array = np.array(self._stream_buffer)
            self._check_drift_and_replace(buffer_array)
            self._stream_buffer.clear()

        return self

    def score_partial(self, X: NDArray[np.float64]) -> float:
        """Score the anomalousness of the next instance using weighted ensemble.

        If no feedback has been provided yet, uses base HST score. After feedback
        starts, blends base HST and ensemble scores over `score_transition_window`
        instances to avoid a discontinuity.

        Args:
            X (np.ndarray): Instance of shape (num_features,).

        Returns:
            float: Anomaly score (higher = more anomalous).
        """
        X = np.asarray(X, dtype=np.float64)

        # Base HST score (negative mass, so higher = more anomalous)
        base_score = self._hst.score_partial(X)

        if not self._feedback_started or self._region_weights is None:
            # No feedback yet: use base HST score
            return base_score

        # Ensemble score using learned weights
        X_feat = self._transform_to_ensemble_features(X.reshape(1, -1))
        ensemble_score = float(X_feat.dot(self._region_weights)[0])

        # Blend scores during transition window to avoid discontinuity
        if (
            self.score_transition_window > 0
            and self._instances_since_feedback < self.score_transition_window
        ):
            alpha = self._instances_since_feedback / self.score_transition_window
            blended_score = (1 - alpha) * base_score + alpha * ensemble_score
            self._instances_since_feedback += 1
            return blended_score

        # After transition, use ensemble score
        return ensemble_score

    def fit_score_partial(self, X: NDArray[np.float64], y: int | None = None) -> float:
        """Score then fit the instance (prequential)."""
        score = self.score_partial(X)
        self.fit_partial(X, y)
        return score

    def should_query(self) -> bool:
        """Return True if the model can request a label (feedback budget not exhausted)."""
        return self._feedback_used < self.feedback_budget

    def request_feedback(self, X: NDArray[np.float64]) -> bool:
        """Determine if the given instance should be queried for a label.

        Simple strategy (QUERY_TYPE=1 in reference): query if score is in top
        tau-quantile of recent scores.

        Note: The reference implementation supports multiple query strategies
        including diversity via compact descriptions (QUERY_TYPE=8). This
        implementation only supports the basic top-tau strategy.
        """
        if not self.should_query():
            return False

        score = self.score_partial(X)
        self._unlabeled_buffer.append(X.copy())
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

    def provide_feedback(self, X: NDArray[np.float64], label: int) -> None:
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

        # Track when feedback starts for score transition
        if not self._feedback_started:
            self._feedback_started = True
            self._instances_since_feedback = 0

        self._labeled_X.append(X.copy())
        self._labeled_y.append(label)
        self._feedback_used += 1

        # Update ensemble weights
        if len(self._labeled_X) > 0 and len(self._unlabeled_buffer) > 0:
            X_labeled = np.array(self._labeled_X)
            y_labeled = np.array(self._labeled_y)
            X_unlabeled = np.array(self._unlabeled_buffer)
            self._update_weights_from_feedback(X_labeled, y_labeled, X_unlabeled)

    def explain_partial(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
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
        self._region_weights = None
        self._uniform_weights = None
        self._region_scores = None
        self._ref_node_dists = None
        self._kl_q_alpha = None
        self._stream_buffer.clear()
        self._drift_ref_initialized = False
        self._feedback_started = False
        self._instances_since_feedback = 0
