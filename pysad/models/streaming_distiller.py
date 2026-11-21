from __future__ import annotations

import math
from typing import ClassVar, Literal

import numpy as np
from scipy.special import expit
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _finite_float, _int_at_least


class _OnlineAutoencoder:
    """Simple online tied-weight denoising autoencoder for StreamingDistiller peers.

    This is a streamlined version of KitNET's _DenoisingAutoencoder with mutual
    distillation support. It learns to reconstruct its input online using
    stochastic gradient descent on tied weights.
    """

    def __init__(
        self,
        n_visible: int,
        lr: float,
        hidden_ratio: float,
        rng: np.random.RandomState,
        distillation_weight: float = 0.5,
    ) -> None:
        n_hidden = max(1, int(math.ceil(n_visible * hidden_ratio)))
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        self.distillation_weight = distillation_weight
        self.n_trained = 0

        # Min-max normalization bounds (updated online)
        self.norm_max = np.full(n_visible, -np.inf, dtype=np.float64)
        self.norm_min = np.full(n_visible, np.inf, dtype=np.float64)

        # Tied weights initialization
        bound = 1.0 / math.sqrt(n_visible)
        self.W = np.asarray(
            rng.uniform(low=-bound, high=bound, size=(n_visible, n_hidden)),
            dtype=np.float64,
        )
        self.hbias = np.zeros(n_hidden, dtype=np.float64)
        self.vbias = np.zeros(n_visible, dtype=np.float64)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using online min-max bounds."""
        scale = self.norm_max - self.norm_min
        # Where scale is 0 (constant feature), use 1.0 to avoid division by zero
        scale = np.where(scale == 0.0, 1.0, scale).astype(np.float64)
        # If norm bounds are still at initial values (no training yet), skip normalization
        not_initialized = ~np.isfinite(self.norm_max) | ~np.isfinite(self.norm_min)
        scale = np.where(not_initialized, 1.0, scale).astype(np.float64)
        norm_min = np.where(not_initialized, 0.0, self.norm_min).astype(np.float64)
        return (x - norm_min) / scale

    def forward(self, x_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass: encode and decode."""
        hidden = expit(x_norm @ self.W + self.hbias)
        reconstructed = expit(hidden @ self.W.T + self.vbias)
        return hidden, reconstructed

    def reconstruction_error(self, x: np.ndarray) -> float:
        """Compute reconstruction error for scoring (no weight update)."""
        x_norm = self.normalize(x)
        _, reconstructed = self.forward(x_norm)
        return float(np.sqrt(np.mean((x_norm - reconstructed) ** 2)))

    def train_step(self, x: np.ndarray, peer_reconstructions: np.ndarray | None = None) -> float:
        """Train on one instance with optional distillation from peer reconstructions.

        Args:
            x: Input instance of shape (n_visible,)
            peer_reconstructions: Optional mean reconstruction from other peers,
                shape (n_visible,). If provided, distillation loss is added.

        Returns:
            Reconstruction error (RMSE) on this instance.
        """
        self.n_trained += 1
        np.maximum(self.norm_max, x, out=self.norm_max)
        np.minimum(self.norm_min, x, out=self.norm_min)

        x_norm = self.normalize(x)
        hidden = expit(x_norm @ self.W + self.hbias)
        reconstructed = expit(hidden @ self.W.T + self.vbias)

        # Reconstruction error
        err_v = x_norm - reconstructed
        err_h = (err_v @ self.W) * hidden * (1.0 - hidden)

        # Distillation gradient: encourage reconstruction to match peer mean
        if peer_reconstructions is not None:
            peer_norm = self.normalize(peer_reconstructions)
            distill_target = peer_norm
            distill_err = reconstructed - distill_target
            # Add distillation gradient to visible error
            err_v = err_v + self.distillation_weight * distill_err
            # Hidden error gets additional term from distillation
            err_h = err_h + self.distillation_weight * (distill_err @ self.W) * hidden * (
                1.0 - hidden
            )

        # Weight updates (tied weights)
        self.W += self.lr * (np.outer(x_norm, err_h) + np.outer(err_v, hidden))
        self.hbias += self.lr * float(err_h.mean())
        self.vbias += self.lr * float(err_v.mean())

        return float(np.sqrt(np.mean((x_norm - reconstructed) ** 2)))

    def get_reconstruction(self, x: np.ndarray) -> np.ndarray:
        """Get the reconstructed output for distillation."""
        x_norm = self.normalize(x)
        _, reconstructed = self.forward(x_norm)
        # Denormalize
        scale = self.norm_max - self.norm_min
        not_initialized = ~np.isfinite(self.norm_max) | ~np.isfinite(self.norm_min)
        scale = np.where(scale == 0.0, 1.0, scale).astype(np.float64)
        scale = np.where(not_initialized, 1.0, scale).astype(np.float64)
        norm_min = np.where(not_initialized, 0.0, self.norm_min).astype(np.float64)
        return reconstructed * scale + norm_min


class StreamingDistiller(BaseModel):
    """Online knowledge distillation for streaming anomaly detection (StreamingDistiller).

    StreamingDistiller maintains an ensemble of online autoencoder peers that learn
    collaboratively through mutual knowledge distillation. Each peer is trained to
    reconstruct the input while also matching the ensemble's consensus reconstruction,
    following the online knowledge distillation paradigm :cite:`hinton2015distilling,chen2020online`.

    The anomaly score combines:
    - Average reconstruction error across peers (how well the input is modeled)
    - Peer disagreement (variance of reconstructions, capturing distillation uncertainty)

    Higher scores indicate more anomalous instances.

    The model uses tied-weight denoising autoencoders as peers with online min-max
    normalization. Peers are updated in a round-robin fashion to maintain diversity.

    Deviations from the cited papers:

    - The original knowledge distillation (Hinton et al.) uses a pre-trained teacher
      and a student. This model uses mutual online distillation among peers (no
      pre-trained teacher), following the online KD paradigm (Chen et al., AAAI 2020).
    - The paper uses classification logits for distillation. This model uses
      reconstruction outputs for distillation, adapted to the unsupervised anomaly
      detection setting.
    - Peers use simple tied-weight autoencoders with sigmoid activations, similar to
      KitNET but with online mutual distillation added.

    Args:
        num_peers (int): Number of peer autoencoders in the ensemble (Default=5).
        hidden_ratio (float): Ratio of hidden to visible neurons for each peer
            (Default=0.5). Must be in (0, 1].
        learning_rate (float): Learning rate for peer weight updates (Default=0.01).
        distillation_weight (float): Weight of distillation loss relative to
            reconstruction loss (Default=0.5). Must be >= 0.
        random_state (int, np.random.RandomState or None): Seed or random number
            generator for peer initialization. None draws from NumPy's global random
            state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    __bibtex__ = """
@inproceedings{hinton2015distilling,
  title={Distilling the knowledge in a neural network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  booktitle={NIPS Deep Learning and Representation Learning Workshop},
  year={2015}
}

@inproceedings{chen2020online,
  title={Online knowledge distillation with diverse peers},
  author={Chen, Defang and Mei, Jian-Ping and Wang, Can and Feng, Yan and Chen, Chun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={04},
  pages={3430--3437},
  year={2020}
}
"""

    _input_kind: ClassVar[Literal["multivariate"]] = "multivariate"

    def __init__(
        self,
        num_peers: int = 5,
        hidden_ratio: float = 0.5,
        learning_rate: float = 0.01,
        distillation_weight: float = 0.5,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.num_peers = _int_at_least(num_peers, "num_peers", 1)
        self.hidden_ratio = _finite_float(hidden_ratio, "hidden_ratio")
        if not 0.0 < self.hidden_ratio <= 1.0:
            raise ValueError(f"hidden_ratio must be in (0, 1], got {self.hidden_ratio}.")
        self.learning_rate = _finite_float(learning_rate, "learning_rate")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}.")
        self.distillation_weight = _finite_float(distillation_weight, "distillation_weight")
        if self.distillation_weight < 0.0:
            raise ValueError(
                f"distillation_weight must be non-negative, got {self.distillation_weight}."
            )

        self.random_state = random_state
        self.peers_: list[_OnlineAutoencoder] = []
        self.num_features_: int = 0
        self.initialized_ = False
        self.n_trained_ = 0

    def _initialize_peers(self, num_features: int) -> None:
        """Initialize peer autoencoders."""
        rng = check_random_state(self.random_state)
        # Use different seeds for each peer
        peer_seeds = [int(rng.randint(2**31)) for _ in range(self.num_peers)]
        self.peers_ = [
            _OnlineAutoencoder(
                n_visible=num_features,
                lr=self.learning_rate,
                hidden_ratio=self.hidden_ratio,
                rng=check_random_state(seed),
                distillation_weight=self.distillation_weight,
            )
            for seed in peer_seeds
        ]
        self.num_features_ = num_features
        self.initialized_ = True

    def _get_peer_reconstructions(self, x: np.ndarray) -> np.ndarray:
        """Get reconstructions from all peers for distillation."""
        reconstructions = np.empty((self.num_peers, self.num_features_), dtype=np.float64)
        for i, peer in enumerate(self.peers_):
            reconstructions[i] = peer.get_reconstruction(x)
        return reconstructions

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> StreamingDistiller:
        """Update all peers with the next instance using mutual distillation.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 1:
            raise ValueError(
                f"StreamingDistiller expects a 1D array of shape ({self.num_features_},), "
                f"got shape {X.shape}."
            )

        if not self.initialized_:
            self._initialize_peers(X.shape[0])

        if X.shape[0] != self.num_features_:
            raise ValueError(
                f"StreamingDistiller expects {self.num_features_} features, got {X.shape[0]}."
            )

        # Get peer reconstructions for distillation (before updating)
        peer_reconstructions = self._get_peer_reconstructions(X)
        peer_mean_reconstruction = peer_reconstructions.mean(axis=0)

        # Train each peer with distillation from the ensemble mean
        for peer in self.peers_:
            peer.train_step(x=X, peer_reconstructions=peer_mean_reconstruction)

        self.n_trained_ += 1
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Score the anomalousness of the next instance.

        The score combines average reconstruction error and peer disagreement.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score.

        Returns:
            float: Anomaly score (higher = more anomalous).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 1:
            raise ValueError(
                f"StreamingDistiller expects a 1D array of shape ({self.num_features_},), "
                f"got shape {X.shape}."
            )

        if not self.initialized_:
            return 0.0

        if X.shape[0] != self.num_features_:
            raise ValueError(
                f"StreamingDistiller expects {self.num_features_} features, got {X.shape[0]}."
            )

        # Compute reconstruction errors and peer reconstructions
        errors = np.empty(self.num_peers, dtype=np.float64)
        reconstructions = np.empty((self.num_peers, self.num_features_), dtype=np.float64)

        for i, peer in enumerate(self.peers_):
            errors[i] = peer.reconstruction_error(X)
            reconstructions[i] = peer.get_reconstruction(X)

        # Average reconstruction error (how well the ensemble models the input)
        mean_error = float(errors.mean())

        # Peer disagreement: variance of reconstructions (distillation uncertainty)
        # High disagreement means peers don't agree on the normal pattern
        peer_var = float(np.mean(np.var(reconstructions, axis=0)))

        # Combine: normalized error + disagreement
        # Normalize error by a running estimate (simple approach: use 1 + mean_error)
        score = mean_error * (1.0 + peer_var)

        return score
