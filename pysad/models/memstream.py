from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel


class MemStream(BaseModel):
    """Detects anomalies with a denoising autoencoder and FIFO memory :cite:`bhatia2022memstream`.

    Implements Algorithm 1 with single-layer ReLU encoder and decoder, an
    encoding dimension of twice the input dimension, Gaussian corruption and
    full-batch Adam training. After training, only the memory and its input
    normalization change. Stored encodings retain their insertion-time
    normalization, as in the authors' implementation.

    The first ``window_size`` instances must be normal: they train the network
    and initialize memory. Scores are zero until this block is complete. This
    replaces the reference script's label-selected initial training set with
    a contiguous stream prefix; labels passed to fit are ignored. Automatic
    retraining is not performed.

    The reference `Stream-AD/MemStream <https://github.com/Stream-AD/MemStream>`_
    (Apache-2.0) uses tanh and a linear decoder; this implementation follows
    the paper's Section 5 ReLU layers instead. It also follows Algorithm 1's
    strict ``score < beta`` rather than the reference's inclusive threshold,
    and supports its discounted K-neighbor score (the reference uses K=1).
    NumPy float64 and its random generator replace PyTorch float32 and its
    generator, so trained weights and scores differ even with the same seed.
    Gaussian noise has standard deviation 0.001, as in the reference.
    Normalization uses sample standard deviations, as in the reference;
    constant features use a scale of one instead of being discarded, so
    changes in a previously constant feature remain observable. A one-record
    memory also uses a scale of one.

    Memory is O(window_size * num_features + num_features ** 2). Each call
    costs O(window_size * num_features + num_features ** 2), independent of
    stream length; the call completing the initial block also trains for
    ``num_epochs`` passes. Default memory is 64 (one of the paper's searched
    sizes), rather than the reference script's 2048.

    Args:
        window_size (int): Number of initial normal instances and memory capacity (Default=64).
        beta (float): Strict upper bound on scores admitted to memory (Default=0.1).
        num_neighbors (int): Number of nearest memory encodings used for scoring; at most window_size (Default=1).
        gamma (float): Neighbor discount factor in [0, 1]; zero uses only the nearest neighbor (Default=0.0).
        learning_rate (float): Adam learning rate (Default=0.01).
        num_epochs (int): Full-batch denoising training passes on the initial block (Default=5000).
        random_state (int, np.random.RandomState or None): Seed or generator for weights and corruption noise. None uses NumPy's global generator (Default=None).
    """

    def __init__(
        self,
        window_size: int = 64,
        beta: float = 0.1,
        num_neighbors: int = 1,
        gamma: float = 0.0,
        learning_rate: float = 0.01,
        num_epochs: int = 5000,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        if not 1 <= num_neighbors <= window_size:
            raise ValueError("num_neighbors must be between 1 and window_size.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1.")
        self.window_size = window_size
        self.beta = beta
        self.num_neighbors = num_neighbors
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.random_state = random_state
        self.num_seen_ = 0
        self.next_ = 0
        weights = gamma ** np.arange(num_neighbors)
        self.neighbor_weights_ = weights / weights.sum()

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> MemStream:
        """Learns from one instance; y is ignored. Returns this model."""
        if self.num_seen_ == 0:
            self.data_ = np.empty((self.window_size, len(X)), dtype=np.float64)
        if self.num_seen_ < self.window_size:
            self.data_[self.num_seen_] = X
            self.num_seen_ += 1
            if self.num_seen_ == self.window_size:
                self._normalize_memory()
                self._train()
                self.memory_ = self._encode(self.data_)
            return self

        encoding = self._encode(X)
        if self._distance(encoding) < self.beta:
            self.data_[self.next_] = X
            self.memory_[self.next_] = encoding
            self.next_ = (self.next_ + 1) % self.window_size
            self._normalize_memory()
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Returns the discounted L1 distance, or zero during initial training."""
        if self.num_seen_ < self.window_size:
            return 0.0
        return self._distance(self._encode(X))

    def _normalize_memory(self) -> None:
        self.mean_ = self.data_.mean(axis=0)
        self.std_ = self.data_.std(axis=0, ddof=int(self.window_size > 1))
        self.std_[self.std_ == 0.0] = 1.0

    def _encode(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(
            ((X - self.mean_) / self.std_).dot(self.encoder_weights_) + self.encoder_bias_, 0.0
        )

    def _distance(self, encoding: np.ndarray) -> float:
        distances = np.abs(self.memory_ - encoding).sum(axis=1)
        if self.num_neighbors == 1 or self.gamma == 0.0:
            return float(distances.min())
        nearest = np.partition(distances, self.num_neighbors - 1)[: self.num_neighbors]
        nearest.sort()
        return float(nearest @ self.neighbor_weights_)

    def _train(self) -> None:
        rng = check_random_state(self.random_state)
        num_features = self.data_.shape[1]
        # Linear-layer initialization matches the reference's uniform bounds.
        bound = 1.0 / np.sqrt(num_features)
        self.encoder_weights_ = rng.uniform(-bound, bound, (num_features, 2 * num_features))
        self.encoder_bias_ = rng.uniform(-bound, bound, 2 * num_features)
        bound = 1.0 / np.sqrt(2 * num_features)
        decoder_weights = rng.uniform(-bound, bound, (2 * num_features, num_features))
        decoder_bias = rng.uniform(-bound, bound, num_features)
        parameters = [self.encoder_weights_, self.encoder_bias_, decoder_weights, decoder_bias]
        first = [np.zeros_like(p) for p in parameters]
        second = [np.zeros_like(p) for p in parameters]
        target = (self.data_ - self.mean_) / self.std_
        for epoch in range(1, self.num_epochs + 1):
            noisy = target + rng.normal(0.0, 0.001, target.shape)
            hidden = np.maximum(noisy.dot(self.encoder_weights_) + self.encoder_bias_, 0.0)
            output = np.maximum(hidden.dot(decoder_weights) + decoder_bias, 0.0)
            delta = (2.0 / target.size) * (output - target) * (output > 0.0)
            hidden_delta = (delta.dot(decoder_weights.T)) * (hidden > 0.0)
            gradients = [
                noisy.T.dot(hidden_delta),
                hidden_delta.sum(axis=0),
                hidden.T.dot(delta),
                delta.sum(axis=0),
            ]
            for parameter, gradient, m, v in zip(
                parameters, gradients, first, second, strict=False
            ):
                m *= 0.9
                m += 0.1 * gradient
                v *= 0.999
                v += 0.001 * gradient * gradient
                parameter -= (
                    self.learning_rate
                    * (m / (1.0 - 0.9**epoch))
                    / (np.sqrt(v / (1.0 - 0.999**epoch)) + 1e-8)
                )
