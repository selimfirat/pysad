from __future__ import annotations

import numpy as np

from pysad.core.base_model import BaseModel


class OnlineOneClassSVM(BaseModel):
    """Scores each instance by its margin under an online one-class support vector machine, the novelty detection variant of NORMA :cite:`kivinen2004online`.

    NORMA runs stochastic gradient descent on the regularized risk in a reproducing kernel Hilbert space. For novelty detection, the hypothesis is the kernel expansion ``f(x) = sum_i alpha_i k(x_i, x)`` with the Gaussian kernel ``k(x_i, x) = exp(-gamma ||x_i - x||^2)``, and the loss is ``max(0, rho - f(x)) - nu * rho``. On each fitted instance ``x_t``, every coefficient decays by ``1 - learning_rate``, and the model updates as in Eq. (20) of the paper, with the signs for ``rho`` that the deviations below explain:

    - If ``f(x_t) < rho``, a margin error, ``x_t`` joins the expansion with ``alpha_t = learning_rate`` and ``rho`` decreases by ``learning_rate * (1 - nu)``.
    - Otherwise, ``alpha_t = 0`` and ``rho`` increases by ``learning_rate * nu``.

    So about a fraction ``nu`` of the instances are margin errors and become support vectors. The score of an instance is ``rho - f(x)``, so that instances far from the recent support vectors get higher scores. The paper ranks instances by ``f(x) - rho``. Before the first margin error, the score is ``rho``.

    The model keeps the terms of the last ``window_size`` instances and drops older terms, the truncation of Fig. 1 of the paper. By Proposition 1, the truncation error falls as ``(1 - learning_rate) ** window_size``, which is about 4e-5 with the defaults. The memory and the time of one call are thus fixed: ``window_size`` instances and ``window_size`` kernel evaluations.

    Deviations from the paper: Eq. (20) prints the opposite signs for the update of ``rho``. With those signs, ``rho`` decreases on every instance that is not a margin error, so no margin error ever occurs and no instance is stored. The signs here are the gradient descent step of Eq. (7) on the loss above. They match the classification update after Eq. (19), and they give the fraction ``nu`` of margin errors that the paper states after Eq. (20). Eq. (20) fixes the regularization constant ``lambda = 1``, as this class does. With ``rho`` starting at 0, another ``lambda`` with the same product ``learning_rate * lambda`` only scales ``f`` and ``rho``, so it gives the same margin errors and the same ranking. The novelty experiment of the paper uses a learning rate that decreases as ``1 / sqrt(t)`` and no truncation. That expansion grows with the stream, so this class uses the constant learning rate and the truncation of Fig. 1 instead. The paper uses ``nu = 0.01``, which flags a small fraction of alerts. The default ``nu`` of 0.5 stores more support vectors, which gives a better ranking of the instances.

    Args:
        nu (float): The fraction of instances that become margin errors and support vectors, in (0, 1] (Default=0.5).
        learning_rate (float): The constant learning rate, eta in the paper, in (0, 1). A larger value forgets old instances faster (Default=0.01).
        gamma (float or None): The inverse width ``1 / (2 * sigma ** 2)`` of the Gaussian kernel. None uses ``2 / num_features``, the width ``2 * sigma ** 2 = 0.5 * num_features`` of the experiment of the paper, whose features lie in [-1, 1] (Default=None).
        window_size (int): The number of most recent instances whose terms the expansion keeps, tau in the paper (Default=1000).
    """

    def __init__(
        self,
        nu: float = 0.5,
        learning_rate: float = 0.01,
        gamma: float | None = None,
        window_size: int = 1000,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        if not (0.0 < nu <= 1.0 and 0.0 < learning_rate < 1.0):
            raise ValueError(
                f"nu must be in (0, 1] and learning_rate in (0, 1), got nu={nu} and learning_rate={learning_rate}."
            )
        self.nu = nu
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.window_size = window_size
        self.rho_ = 0.0
        self.num_seen_ = 0

    def _init_model(self, num_features: int) -> None:
        self.gamma_ = 2.0 / num_features if self.gamma is None else self.gamma
        # Slot t % window_size holds instance t and its coefficient, 0 if it was not a margin error.
        self.support_vectors_ = np.zeros((self.window_size, num_features))
        self.alphas_ = np.zeros(self.window_size)

    def _decision(self, X: np.ndarray) -> float:
        """Returns f(X), the kernel expansion at the instance."""
        if self.num_seen_ == 0:
            return 0.0
        diffs = self.support_vectors_ - X  # (window_size, num_features)
        sq_dists = np.einsum("ij,ij->i", diffs, diffs)  # (window_size,)
        # An elementwise sum: Apple's Accelerate BLAS raises spurious warnings in a matmul here.
        return float((self.alphas_ * np.exp(-self.gamma_ * sq_dists)).sum())

    def _fit(self, X: np.ndarray, decision: float) -> None:
        if self.num_seen_ == 0:
            self._init_model(X.shape[0])
        self.alphas_ *= 1.0 - self.learning_rate
        slot = self.num_seen_ % self.window_size
        if decision < self.rho_:
            self.support_vectors_[slot] = X
            self.alphas_[slot] = self.learning_rate
            self.rho_ -= self.learning_rate * (1.0 - self.nu)
        else:
            self.alphas_[slot] = 0.0
            self.rho_ += self.learning_rate * self.nu
        self.num_seen_ += 1

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> OnlineOneClassSVM:
        """Updates the model with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        self._fit(X, self._decision(X))
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: ``rho - f(X)``, the anomalousness score of the instance.
        """
        return self.rho_ - self._decision(X)

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Scores the next instance, and then updates the model with it.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The anomalousness score of the instance.
        """
        decision = self._decision(X)
        score = self.rho_ - decision
        self._fit(X, decision)
        return score
