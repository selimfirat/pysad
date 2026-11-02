from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.utils import check_random_state

from pysad.core.base_model import BaseModel
from pysad.utils.validation import _finite_float, _int_at_least

# Added to the diagonal of each covariance before it is factorized, as ``reg_covar`` in scikit-learn's GaussianMixture.
COVARIANCE_REGULARIZATION = 1e-6


class SmartSifter(BaseModel):
    """Scores each point by its logarithmic loss under a Gaussian mixture that the SDEM algorithm learns online :cite:`yamanishi2000smartsifter,yamanishi2004smartsifter`.

    The model keeps a mixture of ``num_components`` Gaussians and learns it with sequentially discounting expectation maximization (SDEM, Figure 2 of :cite:`yamanishi2000smartsifter`). Each instance ``x`` takes one E-step and one M-step. The responsibility of component ``i`` is ``(1 - alpha * r) * c_i p_i(x) / sum_l c_l p_l(x) + alpha * r / k``, and the weight, the mean and the covariance of the component decay by ``1 - r`` before they take ``x`` in with this responsibility. A larger ``discount`` ``r`` forgets the past faster, so the model follows a drifting stream faster.

    The score of ``x`` is the logarithmic loss ``-log p(x)`` under the mixture before it learns ``x``, so higher scores are more anomalous.

    The implementation differs from the paper as follows:

    - The model takes continuous features only. The paper also takes categorical features, with one discounted histogram (SDLE) over their cells and one mixture per cell. A PySAD instance is a numeric vector, so the model keeps a single cell.
    - The paper also defines a Hellinger score, the distance between the mixtures before and after learning ``x``. The model does not give it.
    - The paper spreads the initial means uniformly over the data space. The range of a stream is not known before it starts, so the model draws them uniformly from ``[0, 1]`` in each feature, which is the data space of min-max scaled input. The paper does not give the initial covariances. The model starts them at ``I / 12``, the covariance of the uniform distribution on ``[0, 1]`` in each feature.
    - The model updates the means and the covariances directly, in the form ``mu = (1 - w) mu + w x`` and ``Lambda = (1 - w) Lambda + w (1 - w) (x - mu)(x - mu)^T`` with ``w = r * gamma_i / c_i``. In exact arithmetic, this form gives the same values as the discounted sums of Figure 2, and it keeps each covariance positive semi-definite in floating point.
    - The model adds ``1e-6`` to the diagonal of each covariance when it computes a density, so that the density stays finite on constant features.

    Args:
        num_components (int): The number of Gaussians ``k`` in the mixture (Default=2).
        discount (float): The discounting parameter ``r``, in ``(0, 1)``. The paper uses ``0.0002`` on streams of about 100,000 points. The weight of a point decays by ``1 - r`` with each later point (Default=0.01).
        alpha (float): The smoothing parameter of the responsibilities, from 1.0 to 2.0 in the paper. It keeps each component weight above ``alpha * r / k`` (Default=2.0).
        random_state (int, np.random.RandomState or None): Seed or random number generator for the initial means. None draws from NumPy's global random state, which `pysad.utils.fix_seed` seeds (Default=None).
    """

    def __init__(
        self,
        num_components: int = 2,
        discount: float = 0.01,
        alpha: float = 2.0,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.num_components = _int_at_least(num_components, "num_components", 1)
        self.discount = _finite_float(discount, "discount")
        if not 0.0 < self.discount < 1.0:
            raise ValueError(f"discount must be in (0, 1), got {self.discount}.")
        self.alpha = _finite_float(alpha, "alpha")
        # Otherwise some responsibilities are negative.
        if not 0.0 <= self.alpha * self.discount <= 1.0:
            raise ValueError(
                f"alpha * discount must be in [0, 1], got {self.alpha * self.discount}."
            )
        self.random_state = random_state
        self.to_init = True

    def _init_model(self, num_features: int) -> None:
        rng = check_random_state(self.random_state)
        self.weights_ = np.full(self.num_components, 1.0 / self.num_components)
        self.means_ = rng.uniform(size=(self.num_components, num_features))
        self.covariances_ = np.tile(np.eye(num_features) / 12.0, (self.num_components, 1, 1))
        self._factorize()
        self.to_init = False

    def _factorize(self) -> None:
        """Caches the inverse Cholesky factors and the log normalizers of the components."""
        d = self.means_.shape[1]
        # (k, d, d) lower triangular factors of the regularized covariances.
        chol = np.linalg.cholesky(self.covariances_ + COVARIANCE_REGULARIZATION * np.eye(d))
        self._inv_chol = np.linalg.inv(chol)
        diagonals = np.diagonal(chol, axis1=1, axis2=2)  # (k, d)
        self._log_norms = -0.5 * d * np.log(2.0 * np.pi) - np.log(diagonals).sum(axis=1)

    def _log_joint(self, x: np.ndarray) -> np.ndarray:
        """Returns ``log(c_i p_i(x))`` for each component, of shape (k,)."""
        z = np.einsum("kij,kj->ki", self._inv_chol, x - self.means_)  # (k, d)
        return np.log(self.weights_) + self._log_norms - 0.5 * (z * z).sum(axis=1)

    def _update(self, x: np.ndarray, log_joint: np.ndarray) -> None:
        """One SDEM step on ``x``, given the log joint densities under the current mixture."""
        r, k = self.discount, self.num_components
        posteriors = np.exp(log_joint - logsumexp(log_joint))
        gammas = (1.0 - self.alpha * r) * posteriors + self.alpha * r / k
        self.weights_ = (1.0 - r) * self.weights_ + r * gammas
        w = (r * gammas / self.weights_)[:, None]  # (k, 1)
        diffs = x - self.means_  # (k, d)
        self.means_ += w * diffs
        self.covariances_ *= (1.0 - w)[:, :, None]
        self.covariances_ += (w * (1.0 - w))[:, :, None] * diffs[:, :, None] * diffs[:, None, :]
        self._factorize()

    def fit_partial(self, X: np.ndarray, y: int | None = None) -> SmartSifter:
        """Updates the mixture with the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            object: The model itself.
        """
        if self.to_init:
            self._init_model(X.shape[0])

        self._update(X, self._log_joint(X))
        return self

    def score_partial(self, X: np.ndarray) -> float:
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The logarithmic loss ``-log p(X)`` of the instance under the current mixture.
        """
        if self.to_init:
            self._init_model(X.shape[0])

        return -logsumexp(self._log_joint(X))

    def fit_score_partial(self, X: np.ndarray, y: int | None = None) -> float:
        """Scores the next instance, and then updates the mixture with it.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit and score.
            y (int): Ignored since the model is unsupervised (Default=None).

        Returns:
            float: The logarithmic loss ``-log p(X)`` of the instance under the mixture before the update.
        """
        if self.to_init:
            self._init_model(X.shape[0])

        log_joint = self._log_joint(X)
        self._update(X, log_joint)
        return -logsumexp(log_joint)
