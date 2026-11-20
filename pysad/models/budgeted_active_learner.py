from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _finite_float, _int_at_least


class BudgetedActiveLearner(BaseModel):
    r"""Budgeted Active Learner for streaming anomaly detection :cite:`zliobaite2014active`.

    This detector adapts the active learning strategies from Žliobaitė et al. (2014) for
    anomaly detection. It maintains running statistics of the stream to compute
    uncertainty scores (z-scores) and uses a variable threshold that adapts to
    maintain a target anomaly rate (budget).

    The algorithm uses three key ideas from the paper:

    1. **Uncertainty scoring**: Instances are scored by their z-score relative to
       the running mean and standard deviation. Higher z-score means more anomalous.

    2. **Dynamic budget allocation**: The threshold for flagging anomalies adapts
       over time. If the observed anomaly rate exceeds the budget, the threshold
       increases (fewer anomalies flagged). If below budget, the threshold
       decreases (more anomalies flagged).

    3. **Randomized exploration**: Optional Gaussian noise is added to the
       uncertainty score before thresholding, which helps detect concept drift
       occurring away from the current decision boundary.

    The model keeps running estimates of mean and variance using Welford's online
    algorithm, so its memory is fixed and one call costs O(num_features).
    It takes a ``random_state`` for the randomized exploration component.

    Deviations from the paper:

    - The paper addresses active learning for classification with a base classifier
      providing posterior probabilities. This implementation uses running z-scores
      as the uncertainty measure for anomaly detection.
    - The paper's "Selective Sampling" strategy (based on Cesa-Bianchi et al.) is
      not implemented; only the variable uncertainty and randomized variable
      uncertainty strategies are provided.
    - The paper uses a fixed initial period without active learning. This
      implementation uses a ``warmup_period`` during which the threshold is not
      adapted and all instances are used to initialize statistics.

    Args:
        budget (float): Target anomaly rate in (0, 1). The threshold adapts to
            keep the long-term anomaly rate near this value (Default=0.1).
        step (float): Adaptation step size for the threshold in (0, 1). Larger
            values adapt faster but may oscillate (Default=0.01).
        warmup_period (int): Number of initial instances used to initialize
            statistics before threshold adaptation begins. Must be at least 1
            (Default=50).
        strategy (str): Uncertainty strategy. Either "variable" for variable
            threshold adaptation, or "randomized" for variable threshold with
            Gaussian noise added to scores (Default="randomized").
        random_state (int, np.random.RandomState or None): Seed or random number
            generator for randomized exploration. None draws from NumPy's global
            random state, which `pysad.utils.fix_seed` seeds (Default=None).

    Raises:
        ValueError: If ``budget`` not in (0, 1), ``step`` not in (0, 1),
            ``warmup_period`` < 1, or ``strategy`` not recognized.
    """

    __bibtex__ = r"""
@article{zliobaite2014active,
  title={Active learning with drifting streaming data},
  author={Žliobaitė, Indrė and Bifet, Albert and Pfahringer, Bernhard and Holmes, Geoffrey},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  volume={25},
  number={1},
  pages={27--39},
  year={2014},
  publisher={IEEE}
}
"""

    def __init__(
        self,
        budget: float = 0.1,
        step: float = 0.01,
        warmup_period: int = 50,
        strategy: str = "randomized",
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.budget = _finite_float(budget, "budget")
        if not 0.0 < self.budget < 1.0:
            raise ValueError(f"budget must be in (0, 1), got {budget}.")
        self.step = _finite_float(step, "step")
        if not 0.0 < self.step < 1.0:
            raise ValueError(f"step must be in (0, 1), got {step}.")
        self.warmup_period = _int_at_least(warmup_period, "warmup_period", 1)
        if strategy not in ("variable", "randomized"):
            raise ValueError(f"strategy must be 'variable' or 'randomized', got {strategy}.")
        self.strategy = strategy
        self.random_state = random_state
        self._rng: np.random.RandomState | None = (
            None if random_state is None else check_random_state(random_state)
        )

        # Running statistics (Welford's online algorithm)
        self.count_ = 0
        self.mean_: np.ndarray | None = None
        self.m2_: np.ndarray | None = None  # Sum of squared differences

        # Budget management
        self.threshold_ = 3.0  # Initial z-score threshold (roughly 0.3% tail)
        self.anomaly_count_ = 0
        self.post_warmup_count_ = 0

    def _update_stats(self, x: np.ndarray) -> None:
        """Update running mean and variance using Welford's algorithm."""
        self.count_ += 1
        if self.mean_ is None:
            self.mean_ = x.copy()
            self.m2_ = np.zeros_like(x)
        else:
            delta = x - self.mean_
            self.mean_ += delta / self.count_
            delta2 = x - self.mean_
            self.m2_ += delta * delta2

    def _get_std(self) -> np.ndarray:
        """Get standard deviation, avoiding division by zero."""
        if self.count_ < 2 or self.mean_ is None or self.m2_ is None:
            return np.array([1.0])
        variance = self.m2_ / (self.count_ - 1)
        # Replace zero variance with 1 to avoid division by zero
        return np.sqrt(np.maximum(variance, 1e-12))

    def _z_score(self, x: np.ndarray) -> float:
        """Compute the maximum z-score across features."""
        if self.mean_ is None:
            return 0.0
        std = self._get_std()
        z_scores = np.abs(x - self.mean_) / std
        return float(np.max(z_scores))

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> BudgetedActiveLearner:
        """Updates the running statistics and adapts the threshold.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): The model ignores this label (Default=None).

        Returns:
            object: The model itself.
        """
        x = np.asarray(X, dtype=np.float64).ravel()
        if x.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {x.shape}.")

        # Compute z-score before updating stats (prequential)
        z = self._z_score(x)

        # Update running statistics
        self._update_stats(x)

        # Budget management after warmup
        if self.count_ > self.warmup_period:
            self.post_warmup_count_ += 1

            # Determine if this instance would be flagged as anomaly
            if self.strategy == "randomized":
                rng = self._rng if self._rng is not None else check_random_state(None)
                z_noisy = z + rng.normal(0, 1.0)  # Add Gaussian noise
                is_anomaly = z_noisy > self.threshold_
            else:
                is_anomaly = z > self.threshold_

            if is_anomaly:
                self.anomaly_count_ += 1

            # Adapt threshold based on current anomaly rate vs budget
            current_rate = self.anomaly_count_ / self.post_warmup_count_
            if current_rate > self.budget:
                # Too many anomalies, raise threshold
                self.threshold_ *= 1.0 + self.step
            else:
                # Too few anomalies, lower threshold
                self.threshold_ *= 1.0 - self.step

            # Keep threshold in reasonable bounds
            self.threshold_ = max(0.1, min(self.threshold_, 10.0))

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the instance with current estimates, without changing the model.

        The score is the z-score of the instance. For the "randomized" strategy,
        Gaussian noise is added to the score to match the randomized exploration
        behavior. During warmup, this reflects deviation from the running mean.
        After warmup, the threshold adapts but the score reflects the strategy.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: The maximum absolute z-score across features (with noise for
            randomized strategy).
        """
        x = np.asarray(X, dtype=np.float64).ravel()
        if x.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {x.shape}.")
        z = self._z_score(x)
        if self.strategy == "randomized":
            rng = self._rng if self._rng is not None else check_random_state(None)
            z = z + rng.normal(0, 1.0)
        return z
