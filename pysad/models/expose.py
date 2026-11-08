from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class EXPoSE(BaseModel):
    """Scores each instance by its expected similarity to the stream, the inner product of its feature map with the kernel mean embedding of the fitted instances :cite:`schneider2016expose`.

    Random Fourier features (random kitchen sinks) approximate the RBF kernel ``k(x, y) = exp(-gamma * ||x - y||^2)``, so one update and one score cost ``O(num_random_features * num_features)`` time and the model takes constant memory. The feature map of an instance ``x`` is the complex vector ``phi(x) = exp(iZx) / sqrt(num_random_features)`` of the paper (Section 6.1). The random frequencies ``Z`` of shape ``(num_random_features, num_features)`` are drawn once, when the first instance is fitted, from ``N(0, 2 * gamma)``.

    The kernel mean embedding ``w`` is updated in place with the features of each fitted instance as ``w = (1 - rate) * w + rate * phi(x)``. With ``decay=None``, the rate of the ``t``-th instance is ``1 / t`` and ``w`` is the running mean of the features (Proposition 1 of the paper). Otherwise the rate is ``decay`` after the first instance, which is the gradual forgetting of Section 5.1.2, where ``decay`` is the factor gamma of the paper: the weight of the newest instance.

    The paper scores an instance ``z`` by ``<phi(z), w> / ||w||^2`` (Section 5.2), where higher means more normal. This model takes the real part of the inner product, which approximates the mean kernel value, and returns its negative, so that a higher score means more anomalous. Before the first instance is fitted, the score is 0. As in the paper, ``fit_score_partial`` scores an instance before it updates the embedding with it.

    Deviations from the paper: the paper draws ``Z`` from ``N(0, sigma^2)`` for the kernel ``exp(-||x - y||^2 / (2 sigma^2))``, but the Fourier transform of that kernel is the density of ``N(0, 1 / sigma^2)``, which this model uses with ``1 / sigma^2 = 2 * gamma``. The paper tunes the kernel bandwidth on labeled instances of each stream and gives no default; the default ``gamma=1.0`` is a choice of this implementation. The paper calls about 20,000 random features typical; the default of 2,000 is a choice of this implementation that costs a tenth as much per instance. The sliding window variant of Section 5.1.1 is not implemented.

    Args:
        num_random_features (int): The number of random Fourier features, r in the paper (Default=2000).
        gamma (float): The width parameter of the RBF kernel ``exp(-gamma * ||x - y||^2)`` (Default=1.0).
        decay (float or None): The weight of the newest instance in the embedding, gamma in Section 5.1.2 of the paper, in [0, 1). None keeps the running mean of all fitted instances (Default=None).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the random frequencies. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self,
        num_random_features: int = 2000,
        gamma: float = 1.0,
        decay: float | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if decay is not None and not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1) or None, got {decay}.")
        self.num_random_features = num_random_features
        self.gamma = gamma
        self.decay = decay
        self.random_state = random_state
        self.to_init = True

    def _init_model(self, num_features: int) -> None:
        rng = check_random_state(self.random_state)
        self.frequencies_: np.ndarray = rng.normal(
            scale=np.sqrt(2.0 * self.gamma), size=(self.num_random_features, num_features)
        )
        self.embedding_: np.ndarray = np.zeros(self.num_random_features, dtype=np.complex128)
        self.num_seen_ = 0
        self.to_init = False

    def _unscaled_features(self, X: np.ndarray) -> np.ndarray:
        """``exp(iZx)``, the feature map without its factor ``1 / sqrt(num_random_features)``."""
        return np.exp(1j * self.frequencies_.dot(X))

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> EXPoSE:
        """Updates the kernel mean embedding with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: self.
        """
        if self.to_init:
            self._init_model(X.shape[0])

        self.num_seen_ += 1
        rate = 1.0 / self.num_seen_ if self.decay is None or self.num_seen_ == 1 else self.decay
        features = self._unscaled_features(X)
        features *= rate / np.sqrt(self.num_random_features)
        self.embedding_ *= 1.0 - rate
        self.embedding_ += features

        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """
        if self.to_init:
            return 0.0

        # np.vdot conjugates its first argument, as the inner product of the paper does.
        similarity = np.vdot(self._unscaled_features(X), self.embedding_).real / np.sqrt(
            self.num_random_features
        )
        return float(-similarity / np.vdot(self.embedding_, self.embedding_).real)
